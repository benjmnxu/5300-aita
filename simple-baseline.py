import pandas as pd
from score import *

dev = "./data/dev.csv"
train = "./data/train.csv"

def get_majority_class(df: pd.DataFrame) -> str:
    majority_class = df['verdict'].mode().iloc[0]
    return majority_class

class MajorityClassifier:
    def __init__(self, df_dev: pd.DataFrame):
        self.majority_class = get_majority_class(df_dev)

    def predict(self, num_samples: int):
        return [self.majority_class] * num_samples


df_dev = pd.read_csv(dev)
df_train = pd.read_csv(train)

model = MajorityClassifier(df_dev)
y_pred = model.predict(len(df_train))

labels = df_train['verdict'].tolist()

with open("true_labels_dev.txt", "w") as true_file:
    for label in labels:
        true_file.write(f"{label}\n")

with open("predicted_labels_simple.txt", "w") as pred_file:
    for pred in y_pred:
        pred_file.write(f"{pred}\n")

accuracy = get_accuracy(labels, y_pred)
precision = get_weighted_precision(labels, y_pred)
recall = get_weighted_recall(labels, y_pred)
fscore = get_weighted_fscore(labels, y_pred)

print("accuracy", accuracy, "precision", precision, "recall", recall, "fscore:", fscore)
