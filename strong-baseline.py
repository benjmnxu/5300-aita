import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Disable wandb to avoid any issues
os.environ["WANDB_DISABLED"] = "true"

# Load datasets
dev_path = "./data/dev.csv"
train_path = "./data/train.csv"

df_dev = pd.read_csv(dev_path)
df_train = pd.read_csv(train_path)[:200]

# Identify text columns
text_columns = ["body", "title"]  # Text columns to process

# Concatenate text columns into a single text feature
df_train["combined_text"] = df_train[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)
df_dev["combined_text"] = df_dev[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

# Load Hugging Face tokenizer and model
model_name = "distilbert-base-uncased"  # Choose your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model, max_length=128):
    """Generate sentence embeddings using a Hugging Face model."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():  # Disable gradient computation
        outputs = model(**inputs)
    # Use the [CLS] token embedding (first token) as sentence embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Generate embeddings for train and dev datasets
X_train_embeddings = generate_embeddings(df_train["combined_text"].tolist(), tokenizer, hf_model)
X_dev_embeddings = generate_embeddings(df_dev["combined_text"].tolist(), tokenizer, hf_model)

print(X_train_embeddings.shape)

# Target variable
y_train = df_train["verdict"]
y_dev = df_dev["verdict"]

# Train Logistic Regression model using only embeddings
model = LogisticRegression(max_iter=250, multi_class="multinomial", solver="lbfgs")
model.fit(X_train_embeddings, y_train)

# Predict on the dev set
y_pred = model.predict(X_dev_embeddings)

# Calculate accuracy
accuracy = accuracy_score(y_dev, y_pred)

print("Accuracy:", accuracy)
