import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler, LabelEncoder

np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
num_ensembles = 5
C_VALUES = [3.00, 3.05, 3.10, 3.14, 3.20]

class BasinClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_ensemble_member(c_val, model_idx, epochs=25, batch_size=256):
    seed = 42 + int(c_val * 100) + model_idx
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    csv_path = f'files/basin_C{c_val:.2f}.csv'
    df = pd.read_csv(csv_path)
    
    df_clean = df[~df['label_id'].isin([0, 4])].copy()
    
    X_raw = df_clean[['x0', 'vx0']].values
    y_raw = df_clean['label_id'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    x_mid = np.median(X_scaled[:, 0])
    vx_mid = np.median(X_scaled[:, 1])
    test_mask = (X_scaled[:, 0] > x_mid) & (X_scaled[:, 1] > vx_mid)

    X_train, y_train = X_scaled[~test_mask], y_encoded[~test_mask]

    Xs = torch.tensor(X_train, dtype=torch.float32)
    Ys = torch.tensor(y_train, dtype=torch.long)

    dataset = data.TensorDataset(Xs, Ys)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BasinClassifier(input_dim=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(X_train)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [C={c_val:.2f} | Model {model_idx+1}/{num_ensembles}] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    return model

for C in C_VALUES:
    save_dir = f'files/models/{C:.2f}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n================ Training Ensemble for C = {C:.2f} ================")
    for i in range(num_ensembles):
        model = train_ensemble_member(C, i)
        path = os.path.join(save_dir, f'MODEL_{i}.pt')
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")