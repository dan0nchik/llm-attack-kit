import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = "artifacts/textgrad"

# Create a list to collect results
results = []

# Iterate through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Filter rows where 'target' is 'EVALUATION'
        df_eval = df[df["target"] == "EVALUATION"]

        # Check if there's any EVALUATION row
        if not df_eval.empty:
            # Find the row with the maximum validation_acc
            best_row = df_eval.loc[df_eval["validation_acc"].idxmax()]
            results.append(
                {
                    "model_name": filename,
                    "validation_acc": best_row["validation_acc"],
                    "prompt": best_row["prompt"],
                }
            )

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(folder_path, "best_prompts.csv"))
