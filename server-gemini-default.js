const express = require('express');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const {Worker, isMainThread, parentPort, workerData} = require('worker_threads');
const axios = require('axios');
const {
    GoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
} = require("@google/generative-ai");
require('dotenv').config()
const redis = require('redis');
const {VertexAI} = require('@google-cloud/vertexai');
const {GoogleAuth} = require('google-auth-library');
const {OpenAI} = require("openai");

const app = express();

const final_translate_prompt = "Translate all above and create response in json form where the key will be the original value"

// Redis client setup
const redisClient = redis.createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379'
});

redisClient.on('error', (err) => console.log('Redis Client Error', err));

// Connect to Redis (this returns a promise)
const connectRedis = async () => {
    await redisClient.connect();
};

const exponentialBackoff = (retries) => Math.pow(2, (5 - retries)) * 100000; // Exponential backoff in milliseconds


// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname)
    }
});
const upload = multer({storage: storage});

const openAIApiKey = process.env.OPENAI_KEY;
const geminiApiKey = process.env.FLASH_KEY;
const logFile = 'progress.log';
const cancelFile = 'cancel.flag';


// Initialize OpenAI client with API key
const openai = new OpenAI({
    apiKey: openAIApiKey,
});


const gemini_regions = [
    "us-central1",
    "us-east1",
    "us-east4",
    "us-east5",
    "us-south1",
    "us-west1",
    "us-west4",
    "northamerica-northeast1",
    "europe-central2",
    "europe-north1",
    "europe-southwest1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast3",
    "asia-south1",
    "asia-southeast1",
    "australia-southeast1",
    "me-central1",
    "me-central2",
    "me-west1",
]

function getRandomRegion() {
    const randomIndex = Math.floor(Math.random() * gemini_regions.length);
    return gemini_regions[randomIndex];
}

// Initialize Gemini API
const genAI = new GoogleGenerativeAI(geminiApiKey);

async function logProgress(message) {
    await fs.appendFile(logFile, message + '\n');
}

const geminiSafetySettings = [
    {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
    {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
];

async function checkCancellation() {
    try {
        await fs.access(cancelFile);
        await logProgress("Cancellation detected.");
        await fs.unlink(cancelFile);
        process.exit(1)
        return true;
    } catch (error) {
        return false;
    }
}

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function translateToHebrewOpenAI(text, prompt, retries = 10) {
    await checkCancellation();

    try {
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: 'gpt-4o-mini-2024-07-18',
            messages: [
                {
                    role: 'system',
                    content: prompt
                },
                {
                    role: 'user',
                    content: text
                }
            ],
            temperature: 0.7,
            max_tokens: 4000,
            top_p: 1
        }, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${openAIApiKey}`
            }
        });

        const translation = response.data.choices[0].message.content.trim();
        return translation;
    } catch (error) {
        /*await logProgress(`Error: ${error.response?.data?.error?.message || error.message}`);
        throw error;*/

        if (error.message.includes('429') || error.message.includes('500')) {
            if (retries > 0) {
                await logProgress(`OPENAI Quota limit reached. Retrying in 60 seconds... (${retries} retries left)`);
                await new Promise(resolve => setTimeout(resolve, 60000)); // Wait 60 seconds
                return translateToHebrewOpenAI(text, prompt, retries - 1);
            } else {
                // throw new Error(`OpenAI quota limit reached. Max retries exceeded: ${error.message}`);
                await logProgress('OpenAI Quota limit reached. Max retries exceeded. Using original text.');
                return text;
            }
        }

        await logProgress(`Unrecoverable OpenAI error: ${error.response?.data?.error?.message || error.message}. ======= ${error.message} Using original text.`);
        return text; // Return the original text for any other errors
    }
}

async function translateToHebrewGemini(text, prompt, retries = 10) {
    await checkCancellation();
    try {
        const model = genAI.getGenerativeModel({
            model: 'gemini-1.5-flash',
            systemInstruction: prompt,
        });
        const generationConfig = {
            temperature: 1,
            topP: 0.95,
            topK: 64,
            maxOutputTokens: 8192,
            responseMimeType: 'text/plain',
        };

        const chatSession = await model.startChat({
            generationConfig,
            geminiSafetySettings,
            history: [
                {
                    role: 'user',
                    parts: [
                        {text: text},
                    ],
                },
            ],
        });
        const result = await chatSession.sendMessage('');
        const translation = result.response.text().trim();

        return translation;
    } catch (error) {
        if (error.message.includes('429 Too Many Requests') || error.message.includes('Resource has been exhausted')) {
            if (retries > 0) {
                await logProgress(`Gemini Quota limit reached. Retrying in 60 seconds... (${retries} retries left)`);
                await new Promise(resolve => setTimeout(resolve, 60000)); // Wait 30 seconds
                return translateToHebrewGemini(text, prompt, retries - 1);
            } else {
                // throw new Error('Quota limit reached. Max retries exceeded. Falling back to OpenAI.');
                await logProgress('Gemini Quota limit reached. Max retries exceeded. Falling back to OpenAI.');
                return translateToHebrewOpenAI(text, prompt);
            }
        }

        if (error.message.includes('blocked due to SAFETY')) {
            await logProgress('Gemini blocked due to safety concerns. Falling back to OpenAI.');
            return translateToHebrewOpenAI(text, prompt);
        } else {
            await logProgress(`Unknown error, falling back to OpenAI - ${error.message}`);
            return translateToHebrewOpenAI(text, prompt);
        }
    }
}

async function batchTranslateToHebrewGemini(textArray, prompt, retries = 10) {
    await checkCancellation();
    try {
        const model = genAI.getGenerativeModel({
            model: 'gemini-1.5-flash',
            systemInstruction: prompt,
        });
        const generationConfig = {
            temperature: 1,
            topP: 0.95,
            topK: 64,
            maxOutputTokens: 8192,
            responseMimeType: 'text/plain',
        };

        const chatSession = await model.startChat({
            generationConfig,
            geminiSafetySettings,
            history: [
                ...textArray.map(text => ({
                    role: "user",
                    parts: [
                        {text: text},
                    ],
                })),
                {
                    role: 'user',
                    parts: [
                        {text: final_translate_prompt},
                    ],
                },
            ],
        });
        const result = await chatSession.sendMessage('');
        let translation = result.response.text().trim();
        // replace ```json with empty string
        translation = translation.replace("```json","");
        translation = translation.replace("```","");

        // await logProgress("Response from gemini: "+translation)

        return JSON.parse(translation);
    } catch (error) {
        if (error.message.includes('429 Too Many Requests') || error.message.includes('Resource has been exhausted')) {
            if (retries > 0) {
                await logProgress(`Gemini Quota limit reached. Retrying in 60 seconds... (${retries} retries left)`);
                await new Promise(resolve => setTimeout(resolve, 60000)); // Wait 30 seconds
                return batchTranslateToHebrewGemini(textArray, prompt, retries - 1);
            } else {
                // throw new Error('Quota limit reached. Max retries exceeded. Falling back to OpenAI.');
                await logProgress('Gemini Quota limit reached. Max retries exceeded. Falling back to OpenAI.');
                return batchTranslateToHebrewOpenAI(textArray, prompt);
            }
        }

        if (error.message.includes('blocked due to SAFETY')) {
            await logProgress('Gemini blocked due to safety concerns. Falling back to OpenAI.');
            return batchTranslateToHebrewOpenAI(textArray, prompt);
        } else {
            await logProgress(`Unknown error, falling back to OpenAI - ${error.message}`);
            return batchTranslateToHebrewOpenAI(textArray, prompt);
        }
    }
}


async function batchTranslateToHebrewGeminiVertex(textArray, prompt, retries = 10) {
    const region = getRandomRegion(); // Select a random region
    await checkCancellation();

    try {
        const googleAuthOptions = {
            keyFile: process.env.GOOGLE_APPLICATION_CREDENTIALS,  // Replace with your service account key file
            scopes: ['https://www.googleapis.com/auth/cloud-platform'],  // Cloud platform scope
        };
        const auth = new GoogleAuth(googleAuthOptions);

        const vertex_ai = new VertexAI({
            project: process.env.GOOGLE_PROJECT_ID,
            location: region,
            googleAuthOptions: googleAuthOptions
        });
        const model = 'gemini-1.5-flash-001';


        const generativeModel = vertex_ai.preview.getGenerativeModel({
            model: model,
            generationConfig: {
                maxOutputTokens: 8192,
                temperature: 1,
                topP: 0.95,
            },
            safetySettings: [
                {category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH'},
                {category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH'},
                {category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH'},
                {category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH'},
            ],
            systemInstruction: {
                parts: [{
                    text: prompt
                }],
            },
        });

        const text1_1 = {text: `input: Please meet your Rabbie\'s guide inside the Edinburgh Bus Station, Gate J and Gate K, St Andrew Square, Edinburgh, EH1 3DQ\\n\\n(Check in closes 15 minutes prior to departure) output: אנא פגשו את מדריך רבי שלכם בתוך תחנת האוטובוסים של אדינבורו, שער J ושער K, כיכר סנט אנדרו, אדינבורו, EH1 3DQ \\n\\n(הרשמה נסגרת 15 דקות לפני היציאה)`};
        const text2_1 = {text: `input: In the afternoon, you stop in “tropical” Plockton, where you can see palm trees growing alongside Loch Carron and take time to enjoy its sheltered serenity. output: בצהריים, תעצרו ב\"טרופי\" פלוקטון, שם תוכלו לראות עצי דקל גדלים לצד לוך קרון ולקחת זמן ליהנות משלווה מבודדת.`};
        const text3_1 = {text: `input: From here, you travel to Ullapool, situated at the mouth of Loch Broom, a beautiful whitewashed fishing village where you spend your first two nights' stay on tour.\\n* Please note that the tours departing on 28th and 29th September 2019 stay two nights in Inverness instead of Ullapool.\\n output: מכאן, תיסעו לאולהפול, הממוקמת בפתח לוך ברום, כפר דייגים לבן יפהפה בו תבלו את שני הלילות הראשונים שלכם בטיול. \\n* שימו לב שהטיולים היוצאים ב-28 וב-29 בספטמבר 2019 יהיו לילה אחד באינברנס במקום אולהפול.`};

        const req = {
            contents: [
                {
                    role: 'user',
                    parts: [
                        /*text1_1,
                        text2_1,
                        text3_1,*/
                        ...textArray.map(text => ({
                            text: text
                        })),
                        {
                            text: final_translate_prompt
                        }
                    ]
                }
            ]
        };

        const response = await generativeModel.generateContent(req);

        let translation = '';

        // Extract the translation from the response object
        let translationText = response?.response?.candidates?.[0]?.content?.parts?.[0]?.text;
        // console.log("Translation Text:", translationText);

        // replace ```json with empty string
        translationText = translationText.replace("```json","");
        translationText = translationText.replace("```","");

        // await logProgress("Response from vertexai: "+translationText)

        if (translationText) {
            translation = translationText.trim();
            return JSON.parse(translation);
        } else {
            await logProgress("Response not found in vertexai, using gemini")
            return batchTranslateToHebrewOpenAI(textArray, prompt);
        }


    } catch (error) {
        if ((error.message.includes('429') || error.message.includes('500') || error.message.includes('RESOURCE_EXHAUSTED') || error.message.includes('503') || error.message.includes('exception posting request to model'))) {
            if (retries > 0) {
                await logProgress(`Vertex Quota limit reached. Retrying in 60 seconds... (${retries} retries left)`);
                await delay(60000); // Wait 60 seconds
                return batchTranslateToHebrewGeminiVertex(textArray, prompt, retries - 1);
            } else {
                await logProgress('Vertex Quota limit reached. Max retries exceeded. Falling back to Gemini again.');
                return batchTranslateToHebrewGemini(textArray, prompt);
            }
        }

        if (error.message.includes('Unable to authenticate your request')) {
            if (retries > 0) {
                await logProgress(`Vertex unauthorized. Retrying in 120 seconds... (${retries} retries left)`);
                await delay(120000); // Wait 120 seconds
                return batchTranslateToHebrewGeminiVertex(textArray, prompt, retries - 1);
            } else {
                await logProgress('Vertex unauthorized. Max retries exceeded. Falling back to Gemini API.');
                return batchTranslateToHebrewGemini(textArray, prompt);
            }
        }

        if (error.message.includes('blocked due to SAFETY')) {
            await logProgress('Vertex AI blocked due to safety concerns. Falling back to Gemini.');
            return batchTranslateToHebrewOpenAI(textArray, prompt);
        } else {
            await logProgress(error.response);
            await logProgress(`VertexAI translation failed: ${error.message}, falling back to gemini`);
            return batchTranslateToHebrewOpenAI(textArray, prompt);
        }
    }
}

// Function to translate texts and return in JSON format
async function batchTranslateToHebrewOpenAI(textArray, prompt, retries = 10) {
    await checkCancellation();

    await logProgress("OPENAI Request working on it...")

    try {
        // Create messages by concatenating the user messages
        const messages = [
            {
                role: 'system',
                content: prompt
            },
            // Concatenate all user messages into one structure
            ...textArray.map(text => ({
                role: 'user',
                content: text
            })),
            {
                role: 'user',
                content: final_translate_prompt // Include the second prompt
            }
        ];

        // Send the batched request to the OpenAI API
        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini-2024-07-18', // Adjust the model as needed
            messages: messages,
            temperature: 1,
            max_tokens: 16383,
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0,
            response_format: {
                "type": "json_object"
            },
        });

        // Log the response and translations in JSON format
        const jsonResponse = response.choices[0].message.content.trim();
        // await logProgress("JSON Response:", jsonResponse);

        // Parse the result as a JSON object
        const translations = JSON.parse(jsonResponse);
        return translations;

    } catch (error) {
        if (error.message.includes('429') || error.message.includes('500')) {
            if (retries > 0) {
                const delayTime = exponentialBackoff(retries);
                await logProgress(`OPENAI Quota limit reached. Retrying in ${delayTime / 1000} seconds... (${retries} retries left)`);
                await new Promise(resolve => setTimeout(resolve, delayTime)); // Exponential backoff delay
                return batchTranslateToHebrewOpenAI(textArray, prompt, retries - 1);
            } else {
                await logProgress('OpenAI Quota limit reached. Max retries exceeded. Using original text.');
                return textArray; // Return the original texts if retries exceed
            }
        }

        await logProgress(`Unrecoverable OpenAI error: ${error.response?.data?.error?.message || error.message}. Using original texts.`);
        return textArray; // Return original texts for other errors
    }
}

async function translateJson(data, prompt, useGemini = false) {
    const translations = new Map();

    async function traverseAndCollect(obj, path = '') {
        if (Array.isArray(obj)) {
            for (let i = 0; i < obj.length; i++) {
                await traverseAndCollect(obj[i], `${path}[${i}]`);
            }
        } else if (typeof obj === 'object' && obj !== null) {
            for (const [key, value] of Object.entries(obj)) {
                const newPath = path ? `${path}.${key}` : key;
                const isNumeric = (string) => Number.isFinite(+string);
                if (key === 'he' && typeof value === 'string' && value.trim() !== '' && !isNumeric(value)) {
                    try {
                        const cachedTranslation = await redisClient.get(value.trim());

                        if (cachedTranslation) {
                            obj[key] = cachedTranslation;
                        } else {
                            translations.set(newPath, value.trim());
                        }
                    } catch (error) {
                        translations.set(newPath, value.trim());
                    }
                } else if (typeof value === 'object' && value !== null) {
                    await traverseAndCollect(value, newPath);
                }
            }
        }
    }

    async function translateAll() {
        if (translations.size > 0) {
            const textsToTranslate = Array.from(translations.values());
            try {
                let jsonResponse = {};
                if(useGemini){
                    jsonResponse = await batchTranslateToHebrewGeminiVertex(textsToTranslate, prompt);
                }else{
                    jsonResponse = await batchTranslateToHebrewOpenAI(textsToTranslate, prompt);
                }

                for (const [path, originalText] of translations) {
                    let translatedText = jsonResponse[originalText];
                    if (!translatedText) {
                        // If exact match not found, try to reconstruct multi-line text
                        translatedText = originalText.split('\n')
                            .map(line => jsonResponse[line.trim()])
                            .filter(Boolean)
                            .join('\n');
                    }
                    if (translatedText) {
                        await setNestedValue(data, path, translatedText);
                        try {
                            await redisClient.set(originalText.trim(), translatedText);
                        } catch (error) {
                            await logProgress(`Redis caching error for ${path}:`, error);
                        }
                    } else {
                        await logProgress(`No translation found for: ${originalText.slice(0, 50)}, using single line translation`);

                        let singleLineTranslation = '';

                        if(useGemini) {
                            singleLineTranslation = await translateToHebrewGemini(originalText, prompt);
                        }else{
                            singleLineTranslation = await translateToHebrewOpenAI(originalText, prompt);
                        }

                        await setNestedValue(data, path, singleLineTranslation);
                        try {
                            await redisClient.set(originalText.trim(), singleLineTranslation.trim());
                        } catch (error) {
                            await logProgress(`Redis caching error for ${path}:`, error);
                        }
                    }
                }
            } catch (error) {
                await logProgress('Error during translation:', error.message);
            }
        }
    }

    async function setNestedValue(obj, path, value) {
        const keys = path.replace(/\[(\d+)\]/g, '.$1').split('.');
        let current = obj;
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            if (!(key in current)) {
                current[key] = isNaN(keys[i+1]) ? {} : [];
            }
            current = current[key];
        }
        current[keys[keys.length - 1]] = value;
    }

    try {
        await traverseAndCollect(data);
        await translateAll();
    } catch (error) {
        await logProgress('Error in translateJson:', error);
    }

    return data;
}




function createWorker(data, workerId, prompt, useGemini) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(__filename, {
            workerData: {data, workerId, openAIApiKey, geminiApiKey, prompt, useGemini}
        });
        worker.on('message', resolve);
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0)
                reject(new Error(`Worker stopped with exit code ${code}`));
        });
    });
}

if (isMainThread) {
    app.use(express.static('public'));

    app.post('/translate', upload.single('file'), async (req, res) => {
        if (!req.file) {
            return res.status(400).json({error: 'No file uploaded.'});
        }

        try {
            await fs.writeFile(logFile, '');  // Clear log file
            await connectRedis();  // Connect to Redis before starting translation

            const fileContent = await fs.readFile(path.join('uploads', req.file.filename), 'utf8');
            const data = JSON.parse(fileContent);
            const prompt = req.body.prompt || "You will be provided with a sentence in English, and your task is to translate it into Hebrew. Please ensure that the translation is complete, contextually appropriate, and does not include any English content or requests for additional context or clarification. Do not use Hebrew accent marks, and always use the Hebrew plural form wherever possible. Remember not to use  accent marks in any condition";
            const useGemini = req.body.useGemini === 'true';

            const numCPUs = require('os').cpus().length - 4;
            const chunkSize = Math.ceil(data.length / numCPUs);
            const chunks = [];

            for (let i = 0; i < data.length; i += chunkSize) {
                chunks.push(data.slice(i, i + chunkSize));
            }

            const workers = chunks.map((chunk, index) => createWorker(chunk, index, prompt, useGemini));
            const results = await Promise.all(workers);

            const translatedData = results.flat();

            const fileName = path.parse(req.file.originalname).name;
            const translatedFileName = `${fileName}_translated.json`;
            const translatedFilePath = path.join('uploads', translatedFileName);
            await fs.writeFile(translatedFilePath, JSON.stringify(translatedData, null, 2));

            res.json({status: 'success', file: `/downloads/${translatedFileName}`});
        } catch (error) {
            res.status(500).json({status: 'error', message: error.message});
        } finally {
            await redisClient.quit();  // Close Redis connection
        }
    });

    app.post('/cancel', async (req, res) => {
        await fs.writeFile(cancelFile, '1');
        res.json({status: 'canceled'});
    });

    app.get('/log', async (req, res) => {
        try {
            const log = await fs.readFile(logFile, 'utf8');
            res.send(log);
        } catch (error) {
            res.status(404).send('No log available.');
        }
    });

    // Serve translated files
    app.use('/downloads', express.static('uploads', {
        setHeaders: (res, path, stat) => {
            res.set('Content-Disposition', 'attachment');
        }
    }));

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
} else {
    // This code will run in worker threads
    async function workerTranslateJson(data, prompt, workerId, useGemini) {
        await connectRedis();  // Connect to Redis in each worker
        try {
            return await Promise.all(data.map(async (item, index) => {
                await logProgress(`===Worker ${workerId}: Processing product ${index + 1} of ${data.length}===`);
                return await translateJson(item, prompt, useGemini);
            }));
        } finally {
            await redisClient.quit();  // Close Redis connection
        }
    }

    (async () => {
        try {
            const result = await workerTranslateJson(workerData.data, workerData.prompt, workerData.workerId, workerData.useGemini);
            parentPort.postMessage(result);
        } catch (error) {
            parentPort.postMessage({error: error.message});
        }
    })();
}