# train_optimized_lstm_pytorch.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -----------------------------
# 1️⃣ SET RANDOM SEEDS
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# 2️⃣ LOAD DATA
# -----------------------------
df = pd.read_csv("output1.csv")

# Fill missing values
df['Protocol'] = df['Protocol'].fillna(0)
df['Length'] = df['Length'].fillna(0)
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# -----------------------------
# 3️⃣ FEATURE ENGINEERING
# -----------------------------
df['packet_count'] = 1
df['avg_size'] = df['Length']
df['size_variation'] = df['Length'].diff().fillna(0)
df['packet_rate'] = df['Length'].rolling(2).sum().fillna(0)
df['rate_change'] = df['packet_rate'].diff().fillna(0)

features = ['packet_count', 'avg_size', 'size_variation', 'packet_rate', 'rate_change']

# -----------------------------
# 4️⃣ SMART LABELING (BALANCED)
# -----------------------------
q1 = df['packet_rate'].quantile(0.33)
q2 = df['packet_rate'].quantile(0.66)

def label(row):
    rate = row['packet_rate']
    if rate <= q1:
        return 0
    elif rate <= q2:
        return 1
    else:
        return 2

df['label'] = df.apply(label, axis=1)
print("\n📊 Class Distribution:")
print(df['label'].value_counts())

# -----------------------------
# 5️⃣ CREATE SEQUENCES
# -----------------------------
TIMESTEPS = 10
X_seq, y_seq = [], []

data = df[features].values
labels = df['label'].values

for i in range(len(data) - TIMESTEPS):
    X_seq.append(data[i:i+TIMESTEPS])
    y_seq.append(labels[i+TIMESTEPS])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"\nSequence shape: {X_seq.shape}")

# -----------------------------
# 6️⃣ TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=SEED, stratify=y_seq
)

# -----------------------------
# 7️⃣ SCALE FEATURES
# -----------------------------
nsamples, ntimesteps, nfeatures = X_train.shape
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, nfeatures)).reshape(nsamples, ntimesteps, nfeatures)
X_test_scaled = scaler.transform(X_test.reshape(-1, nfeatures)).reshape(X_test.shape[0], ntimesteps, nfeatures)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# 8️⃣ CUSTOM DATASET
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 9️⃣ HANDLE CLASS IMBALANCE
# -----------------------------
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
weights = class_weights[y_train]
weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

# -----------------------------
# 🔟 LSTM MODEL
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim//2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = LSTMClassifier(nfeatures)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 1️⃣1️⃣ TRAIN LOOP
# -----------------------------
EPOCHS = 15  # reduced for speed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

# -----------------------------
# 1️⃣2️⃣ EVALUATION
# -----------------------------
model.eval()
y_pred_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        y_pred_list.extend(torch.argmax(preds, axis=1).cpu().numpy())

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_list))

cm = confusion_matrix(y_test, y_pred_list)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 1️⃣3️⃣ SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "tcp_udp_lstm_pytorch.pt")
print("🚀 Model saved successfully!")