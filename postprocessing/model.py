import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

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

EPOCHS = 150
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
PATIENCE = 30

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_features=256, scale=5.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, num_features, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        z = x @ self.B
        return torch.cat([torch.sin(z), torch.cos(z), x], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class BasinClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=3, hidden=256, n_blocks=3, 
                 ff_dim=256, ff_scale=5.0, dropout=0.1, ff_seed=0):
        super().__init__()
        self.ff = FourierFeatures(in_dim=input_dim, num_features=ff_dim, 
                                  scale=ff_scale, seed=ff_seed)
        in_dim = 2 * ff_dim + input_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.ff(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        log_pt = -F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(log_pt)
        
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * (-log_pt)
        
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            loss = self.alpha[target] * loss
            
        return loss.mean()

def create_fixed_spatial_split(df, seed, n_bins=20, test_frac=0.15, val_frac=0.15):
    x_bins = pd.qcut(df['x0'], q=n_bins, labels=False, duplicates='drop')
    vx_bins = pd.qcut(df['vx0'], q=n_bins, labels=False, duplicates='drop')
    
    max_x_bin = x_bins.max() + 1
    df['block_id'] = x_bins * max_x_bin + vx_bins
    
    rng = np.random.RandomState(seed)
    unique_blocks = df['block_id'].unique()
    
    n_test = max(1, int(len(unique_blocks) * test_frac))
    test_blocks = rng.choice(unique_blocks, size=n_test, replace=False)
    
    remaining_blocks = np.setdiff1d(unique_blocks, test_blocks)
    n_val = max(1, int(len(remaining_blocks) * (val_frac / (1.0 - test_frac))))
    val_blocks = rng.choice(remaining_blocks, size=n_val, replace=False)
    
    test_mask = df['block_id'].isin(test_blocks).values
    val_mask = df['block_id'].isin(val_blocks).values
    train_mask = ~(test_mask | val_mask)
    
    return train_mask, val_mask, test_mask

def train_single_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                       epochs, patience, device, model_idx, c_val):
    best_val_ba = -1.0
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            bs = inputs.size(0)
            running_loss += loss.item() * bs
            samples += bs
            
        scheduler.step()
        train_loss = running_loss / samples
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        val_ba = balanced_accuracy_score(all_targets, all_preds)
        
        if val_ba > best_val_ba + 1e-4:
            best_val_ba = val_ba
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [Model {model_idx+1}] Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | Val BalAcc: {val_ba:.4f}")
            
        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (Best Val BalAcc: {best_val_ba:.4f})")
            break
            
    model.load_state_dict(best_state)
    return model, best_val_ba

def evaluate_ensemble(models, test_loader, label_encoder, device):
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_probs = []
            for model in models:
                model.eval()
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.cpu().numpy())
            all_probs.append(np.stack(batch_probs, axis=1)) # (batch, num_models, num_classes)
            all_targets.extend(labels.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0) # (N, M, C)
    all_targets = np.array(all_targets)
    
    mean_probs = all_probs.mean(axis=1) # (N, C)
    preds = mean_probs.argmax(axis=1)
    
    ba = balanced_accuracy_score(all_targets, preds)
    macro_f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
    
    print("\n  Ensemble Classification Report:")
    print(classification_report(
        all_targets, preds, 
        target_names=[f"Class {c}" for c in label_encoder.classes_],
        zero_division=0
    ))
    print("  Ensemble Confusion Matrix:")
    print(confusion_matrix(all_targets, preds))
    print(f"  Ensemble Balanced Accuracy: {ba:.4f}")
    print(f"  Ensemble Macro F1: {macro_f1:.4f}")
    
    eps = 1e-12
    H_mean = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    H_individual = -np.sum(all_probs * np.log(all_probs + eps), axis=2) # (N, M)
    mean_H_individual = H_individual.mean(axis=1)
    
    epistemic_uncertainty = H_mean - mean_H_individual
    
    return preds, all_targets, mean_probs, epistemic_uncertainty, ba

for C in C_VALUES:
    print(f"\n{'='*60}")
    print(f"Training Ensemble for C = {C:.2f}")
    print(f"{'='*60}")
    
    csv_path = f"files/data/basin_C{C:.2f}.csv"
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}, skipping.")
        continue
        
    df = pd.read_csv(csv_path)
    df_clean = df[~df["label_id"].isin([0, 4])].copy()
    
    split_seed = 42 + int(C * 100)
    train_mask, val_mask, test_mask = create_fixed_spatial_split(df_clean, seed=split_seed)
    
    X_raw = df_clean[["x0", "vx0"]].values
    y_raw = df_clean["label_id"].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    
    X_train_raw, y_train = X_raw[train_mask], y_encoded[train_mask]
    X_val_raw, y_val = X_raw[val_mask], y_encoded[val_mask]
    X_test_raw, y_test = X_raw[test_mask], y_encoded[test_mask]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"  Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    train_dataset = data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_counts = np.bincount(y_train, minlength=num_classes)
    raw_weights = len(y_train) / (num_classes * class_counts)
    raw_weights = np.clip(raw_weights, 0, 5.0)
    class_weights = raw_weights / np.mean(raw_weights)
    print(f"  Class weights (capped inverse freq): {class_weights}")
    
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)
    
    models = []
    save_dir = f"files/models/{C:.2f}"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_ensembles):
        print(f"\n  --- Training Model {i+1}/{num_ensembles} ---")
        model_seed = split_seed + i + 1
        torch.manual_seed(model_seed)
        
        model = BasinClassifier(
            input_dim=2, num_classes=num_classes, hidden=256, n_blocks=3,
            ff_dim=256, ff_scale=5.0, ff_seed=model_seed
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        model, best_ba = train_single_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            EPOCHS, PATIENCE, device, i, C
        )
        models.append(model)
        
        torch.save(model.state_dict(), os.path.join(save_dir, f"MODEL_{i}.pt"))
        
    torch.save({
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "label_classes": label_encoder.classes_,
    }, os.path.join(save_dir, "preprocessing.pt"))
    
    print(f"\n  --- Evaluating Ensemble on Test Set ---")
    preds, targets, mean_probs, epi_unc, ens_ba = evaluate_ensemble(
        models, test_loader, label_encoder, device
    )
    
    print(f"  Mean Epistemic Uncertainty: {epi_unc.mean():.4f}")
    print(f"  Max Epistemic Uncertainty:  {epi_unc.max():.4f}")