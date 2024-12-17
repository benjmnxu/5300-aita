import pandas as pd
import os

# Path to the CSV file and output directory
csv_path = "data/predicted_dev_1501-2250.csv"
output_path = "labels/back_dev_predicted.txt"

# Read the CSV file and write the 'target' column to the output file
data = pd.read_csv(csv_path)
data["target"].to_csv(output_path, index=False, header=False)
