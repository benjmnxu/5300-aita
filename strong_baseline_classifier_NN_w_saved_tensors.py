import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import pandas as pd


df_train = pd.read_csv('data/train.csv') 
df_dev = pd.read_csv('data/dev.csv') 

y_train = df_train["target"]
y_dev = df_dev["target"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_dev_encoded = label_encoder.transform(y_dev)


label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

data = torch.load('data/tensors.pt')
X_train_tensor = data['X_train']
y_train_tensor = data['y_train']
X_dev_tensor = data['X_dev']
y_dev_tensor = data['y_dev']

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

input_size = X_train_tensor.shape[1]
num_classes = len(torch.unique(y_train_tensor))
model = NeuralNetwork(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
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

y_dev_pred_labels = [reverse_label_mapping[pred] for pred in y_dev_pred]

accuracy = accuracy_score(y_dev, y_dev_pred_labels)
print("Accuracy on the dev set:", accuracy)

with open("predicted_labels_NN.txt", "w") as pred_file:
    for pred in y_dev_pred_labels:
        pred_file.write(f"{pred}\n")