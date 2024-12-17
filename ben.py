import pandas as pd

train_path = "data/train.csv"

data = pd.read_csv(train_path)

# Print total number of target entries
print("Total entries in target:", len(data["target"]))

# Print number of unique classes in target
print("Number of unique classes in target:", data["target"].nunique())

print("Number of unique classes in target:", data["target"].unique())

print(data.iloc[0])