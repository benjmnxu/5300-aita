import pandas as pd
from evaluate import *

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

accuracy = get_accuracy(labels, y_pred)
precision = get_weighted_precision(labels, y_pred)
recall = get_weighted_recall(labels, y_pred)
fscore = get_weighted_fscore(labels, y_pred)

print("accuracy", accuracy, "precision", precision, "recall", recall, "fscore:", fscore)
