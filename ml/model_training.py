import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# === Load and Prepare Data ===
df = pd.read_csv('data/updated_dataset_new.csv')
df.columns = df.columns.str.strip()

# Compute pH from min and max if present
if 'pH_min' in df.columns and 'pH_max' in df.columns:
    df['pH'] = df[['pH_min', 'pH_max']].mean(axis=1)

# Define feature columns (9 total)
FEATURES = [
    'Hardness', 'Solids_TDS', 'Sulphate', 'Chloramine', 'Conductivity',
    'Organic_Carbon', 'Trihalomethane', 'Turbidity', 'pH'
]

# Handle missing values
X = df[FEATURES].fillna(df[FEATURES].mean())
y = df['Portability'].astype(int).values.reshape(-1, 1)

# === Compute Medians for Missing Sensors ===
feature_medians = X.median().to_dict()
os.makedirs('models', exist_ok=True)
joblib.dump(feature_medians, 'models/feature_medians.pkl')

# === Normalize Features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# === Define Neural Network ===
class WaterQualityNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Initialize Model, Loss, Optimizer ===
model = WaterQualityNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Training Loop ===
print("Training started...")
for epoch in range(100):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    preds = (outputs > 0.5).float()
    correct = (preds == y_train_tensor).float().sum()
    accuracy = correct / y_train_tensor.numel()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")

# === Save Model and Scaler ===
torch.save(model.state_dict(), 'models/model.pth')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model, scaler, and medians saved successfully.")
