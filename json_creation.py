import os
import csv
import json

# Path where all CSVs are stored
CSV_DIR = './hardware_csvs'
OUTPUT_JSON = './hardware_components.json'

def snake_case(filename):
    return filename.replace(".csv", "").lower()

def load_all_csvs(csv_dir):
    db = {}
    for file in os.listdir(csv_dir):
        if file.endswith('.csv'):
            category = snake_case(file)
            file_path = os.path.join(csv_dir, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                items = [dict(row) for row in reader]
                db[category] = items
    return db

def save_as_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    db = load_all_csvs(CSV_DIR)
    save_as_json(db, OUTPUT_JSON)
    print(f"Structured JSON saved to {OUTPUT_JSON}")