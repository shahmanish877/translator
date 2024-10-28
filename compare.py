import json

# Load both JSON files with explicit encoding
with open('triper 10k products_translated.json', 'r', encoding='utf-8') as translated_file:
    translated_data = json.load(translated_file)

with open('triper 10k products.json', 'r', encoding='utf-8') as original_file:
    original_data = json.load(original_file)

# Extract _id and productCode pairs from both datasets
translated_ids = {(item.get('_id'), item.get('productCode')) for item in translated_data}
original_ids = {(item.get('_id'), item.get('productCode')) for item in original_data}

# Find missing pairs in the translated data
missing_in_translated = original_ids - translated_ids

# Output the results
print(f"Number of missing items in translated JSON: {len(missing_in_translated)}")
print("Missing _id and productCode pairs:")
for missing_pair in missing_in_translated:
    print(missing_pair)
