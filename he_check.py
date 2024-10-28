import json
import re

def is_english(text):
    # This regex pattern matches strings that contain only English letters, numbers, and common punctuation
    english_pattern = re.compile(r'^[a-zA-Z0-9\s.,!?"-]+$')
    return bool(english_pattern.match(text))

def check_he_field(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and 'he' in value:
                he_value = value.get('he')
                if he_value and isinstance(he_value, str) and is_english(he_value):
                    print(f'English text found in "he" field: {he_value}')
            else:
                check_he_field(value)
    elif isinstance(data, list):
        for item in data:
            check_he_field(item)

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON in file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
    return None

if __name__ == "__main__":
    json_file_path = 'products-limit-10000-offset-10000_translated.json'
    data = load_json_file(json_file_path)
    if data:
        check_he_field(data)
    else:
        print("Unable to process the JSON data.")