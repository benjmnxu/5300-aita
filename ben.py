import pandas as pd

# File paths
# dev_path = "data/dev.csv"
# train_path = "data/train.csv"
# test_path = "data/test.csv"
# synth_esh = "data/synthetic_ESH_1800.csv"
# synth_nah = "data/synthetic_NAH_1800.csv"
# synth_yta = "data/synthetic_YTA_1800.csv"

# # Load data
# dev_data = pd.read_csv(dev_path)
# test_data = pd.read_csv(test_path)
# train_data = pd.read_csv(train_path)

# synth_esh_data = pd.read_csv(synth_esh)
# synth_nah_data = pd.read_csv(synth_nah)
# synth_yta_data = pd.read_csv(synth_yta)

# # Add synthetic data to dev set (first 240 rows)
# dev_data = pd.concat([
#     dev_data,
#     synth_esh_data.iloc[:240],
#     synth_nah_data.iloc[:240],
#     synth_yta_data.iloc[:240]
# ], ignore_index=True)

# # Add synthetic data to test set (240 to 479 rows)
# test_data = pd.concat([
#     test_data,
#     synth_esh_data.iloc[240:480],
#     synth_nah_data.iloc[240:480],
#     synth_yta_data.iloc[240:480]
# ], ignore_index=True)

# # Add synthetic data to train set (remaining rows)
# train_data = pd.concat([
#     train_data,
#     synth_esh_data.iloc[480:],
#     synth_nah_data.iloc[480:],
#     synth_yta_data.iloc[480:]
# ], ignore_index=True)

# # Save new datasets
# dev_data.to_csv("data/new_dev.csv", index=False)
# test_data.to_csv("data/new_test.csv", index=False)
# train_data.to_csv("data/new_train.csv", index=False)

# Load new_dev_data
new_dev_data = pd.read_csv("data/new_train.csv")

# Get counts of each target label
target_counts = new_dev_data['target'].value_counts()

# Display specific counts
nta_count = target_counts.get('NTA', 0)
yta_count = target_counts.get('YTA', 0)
nah_count = target_counts.get('NAH', 0)
esh_count = target_counts.get('ESH', 0)

# Print results
print(f"NTA: {nta_count}")
print(f"YTA: {yta_count}")
print(f"NAH: {nah_count}")
print(f"ESH: {esh_count}")

