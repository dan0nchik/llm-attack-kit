import os
import json
import csv

# Define the folder containing the JSON files
folder_path = "artifacts/textgrad"

# Iterate through all .json files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        json_path = os.path.join(folder_path, filename)

        # Load the JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # Extract columns and data
        columns = content.get("columns", [])
        data = content.get("data", [])

        # Write to CSV file with the same name
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(folder_path, csv_filename)

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # write header
            writer.writerows(data)  # write data rows

        print(f"Converted {filename} to {csv_filename}")
