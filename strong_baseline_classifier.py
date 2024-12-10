import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_train = pd.read_csv('data/balanced_train_dataset_augmented.csv') 
df_dev = pd.read_csv('data/balanced_dev_dataset_augmented.csv') 

model_name = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name).to(device) 

def standardize_labels(df):
    df["target"] = df["target"].str.lower().str.replace("-", " ").str.strip()
    return df

df_train = standardize_labels(df_train)
df_dev = standardize_labels(df_dev)

def generate_batched_embeddings(texts, tokenizer, model, batch_size=512, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Generating embeddings'):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)  
        with torch.no_grad(): 
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


X_train_body_embeddings = generate_batched_embeddings(df_train["body"].tolist(), tokenizer, hf_model, batch_size=64)
X_dev_body_embeddings = generate_batched_embeddings(df_dev["body"].tolist(), tokenizer, hf_model, batch_size=64)

X_train_title_embeddings = generate_batched_embeddings(df_train["title"].tolist(), tokenizer, hf_model, batch_size=64)
X_dev_title_embeddings = generate_batched_embeddings(df_dev["title"].tolist(), tokenizer, hf_model, batch_size=64)

X_train_embeddings = np.hstack((X_train_body_embeddings, X_train_title_embeddings))
X_dev_embeddings = np.hstack((X_dev_body_embeddings, X_dev_title_embeddings))

scaler = StandardScaler()
X_train_embeddings_scaled = scaler.fit_transform(X_train_embeddings)
X_dev_embeddings_scaled = scaler.transform(X_dev_embeddings)

print("Train embeddings shape:", X_train_embeddings.shape)
print("Dev embeddings shape:", X_dev_embeddings.shape)

y_train = df_train["target"]
y_dev = df_dev["target"]

pca = PCA(n_components=30) # 30 worked best after quasi grid-search 
X_train_pca = pca.fit_transform(X_train_embeddings)
X_dev_pca = pca.transform(X_dev_embeddings)
model = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs")
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_dev_pca)
accuracy = accuracy_score(y_dev, y_pred)

print("Accuracy:", accuracy)

with open("predicted_labels_strong_augmented.txt", "w") as pred_file:
    for pred in y_pred:
        pred_file.write(f"{pred}\n")

with open("true_labels_dev_strong_augmented.txt", "w") as dev_file:
    for d in y_dev:
        dev_file.write(f"{d}\n")

