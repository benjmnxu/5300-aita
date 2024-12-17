import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_train = pd.read_csv('data/new_train.csv') 
df_dev = pd.read_csv('data/new_dev.csv') 

model_name = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name).to(device) 

print(len(df_train))
print(len(df_dev))

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


X_train_title_embeddings = generate_batched_embeddings(df_train["title"].tolist(), tokenizer, hf_model, batch_size=64)
X_dev_title_embeddings = generate_batched_embeddings(df_dev["title"].tolist(), tokenizer, hf_model, batch_size=64)

X_train_body_embeddings = generate_batched_embeddings(df_train["body"].tolist(), tokenizer, hf_model, batch_size=64)
X_dev_body_embeddings = generate_batched_embeddings(df_dev["body"].tolist(), tokenizer, hf_model, batch_size=64)
X_train_embeddings = np.hstack((X_train_title_embeddings, X_train_body_embeddings))
X_dev_embeddings = np.hstack((X_dev_title_embeddings, X_dev_body_embeddings))

y_train = df_train["target"]
y_dev = df_dev["target"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_dev_encoded = label_encoder.transform(y_dev)

print("Train embeddings shape:", X_train_embeddings.shape)
print("Dev embeddings shape:", X_dev_embeddings.shape)

scaler = StandardScaler()
X_train_embeddings_scaled = scaler.fit_transform(X_train_embeddings)
X_dev_embeddings_scaled = scaler.transform(X_dev_embeddings)

pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train_embeddings_scaled)
X_dev_pca = pca.transform(X_dev_embeddings_scaled)

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_dev_tensor = torch.tensor(X_dev_pca, dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev_encoded, dtype=torch.long)

torch.save({
    'X_train': X_train_tensor,
    'y_train': y_train_tensor,
    'X_dev': X_dev_tensor,
    'y_dev': y_dev_tensor
}, 'data/synthetic_tensors.pt')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_size = X_train_pca.shape[1]
num_classes = len(set(y_train)) 
model = NeuralNetwork(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_dev_pred_logits = model(X_dev_tensor)
    y_dev_pred = torch.argmax(y_dev_pred_logits, dim=1).numpy()


label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

y_dev_pred_labels = [reverse_label_mapping[pred] for pred in y_dev_pred]

accuracy = accuracy_score(y_dev, y_dev_pred_labels)
print("Accuracy on the dev set:", accuracy)

with open("predicted_labels_logistic_synthetic.txt", "w") as pred_file:
    for pred in y_dev_pred_labels:
        pred_file.write(f"{pred}\n")
