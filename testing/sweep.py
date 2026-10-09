import os
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
C_VAL = 3.10
RES = 700
DATA_PATH = f"files/data/{C_VAL:.2f}/{RES}x{RES}.csv"
OUT_DIR = "files/sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "sweep_summary.csv")

FF_SCALES = [2.5, 5.0, 10.0]
HIDDEN_DIMS = [256, 512, 1024]
TRAIN_SEEDS = [42, 123, 456]
FF_SEED = 0

EPOCHS = 100
BATCH_SIZE = 1024
LR = 3e-4
PATIENCE = 20

EVAL_BINS = 20
EVAL_FRAC = 0.15
VAL_FRAC = 0.15
SPLIT_SEED = 42
LABEL_CLASSES = np.array([1, 2, 3])

EPSILONS = np.logspace(np.log10(0.005), np.log10(0.05), 8)
N_PERTURB_SAMPLES = 2000

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
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))

class BasinClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=3, hidden=256, n_blocks=3,
                 ff_dim=256, ff_scale=5.0, dropout=0.1, ff_seed=0):
        super().__init__()
        self.ff = FourierFeatures(input_dim, ff_dim, ff_scale, ff_seed)
        in_dim = 2 * ff_dim + input_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(self.ff(x))))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        log_pt = -F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(log_pt)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * (-log_pt)
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            loss = self.alpha[target] * loss
        return loss.mean()

def get_fixed_block_split():
    rng = np.random.RandomState(SPLIT_SEED)
    all_blocks = np.arange(EVAL_BINS * EVAL_BINS)
    n_eval = max(1, int(len(all_blocks) * EVAL_FRAC))
    eval_blocks = rng.choice(all_blocks, size=n_eval, replace=False)
    remaining = np.setdiff1d(all_blocks, eval_blocks)
    n_val = max(1, int(len(remaining) * (VAL_FRAC / (1.0 - EVAL_FRAC))))
    val_blocks = rng.choice(remaining, size=n_val, replace=False)
    return set(eval_blocks.tolist()), set(val_blocks.tolist())

def assign_block_ids(df):
    x_edges = np.linspace(-1.5, 1.5, EVAL_BINS + 1)
    vx_edges = np.linspace(-3.0, 3.0, EVAL_BINS + 1)
    xb = np.clip(np.digitize(df["x0"].values, x_edges) - 1, 0, EVAL_BINS - 1)
    vxb = np.clip(np.digitize(df["vx0"].values, vx_edges) - 1, 0, EVAL_BINS - 1)
    return xb * EVAL_BINS + vxb

EVAL_BLOCKS, VAL_BLOCKS = get_fixed_block_split()

df = pd.read_csv(DATA_PATH)
df["block_id"] = assign_block_ids(df)
df = df[~df["label_id"].isin([0, 4])].copy()

label_encoder = LabelEncoder()
label_encoder.fit(LABEL_CLASSES)
num_classes = len(label_encoder.classes_)

eval_mask = df["block_id"].isin(EVAL_BLOCKS)
val_mask = df["block_id"].isin(VAL_BLOCKS)
train_mask = ~(eval_mask | val_mask)

df_train = df[train_mask]
df_val = df[val_mask]
df_test = df[eval_mask]

X_train_raw = df_train[["x0", "vx0"]].values
y_train = label_encoder.transform(df_train["label_id"].values)

y_val = label_encoder.transform(df_val["label_id"].values)
y_test = label_encoder.transform(df_test["label_id"].values)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(df_val[["x0", "vx0"]].values)
X_test = scaler.transform(df_test[["x0", "vx0"]].values)

class_counts = np.bincount(y_train, minlength=num_classes)
safe_counts = np.maximum(class_counts, 1)
raw_weights = len(y_train) / (num_classes * safe_counts)
raw_weights = np.clip(raw_weights, 0.0, 5.0)
class_weights = raw_weights / np.mean(raw_weights)

train_dataset = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = FocalLoss(alpha=class_weights, gamma=1.0)

def train_model(model, seed):
    torch.manual_seed(seed)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_ba = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                preds = model(inputs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
        
        val_ba = balanced_accuracy_score(all_targets, all_preds)
        if val_ba > best_val_ba + 1e-4:
            best_val_ba = val_ba
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break
            
    model.load_state_dict(best_state)
    return model, best_val_ba

def compute_beta_metrics(models, scaler):
    full_df = pd.read_csv(DATA_PATH)
    phys_df = full_df[~full_df["label_id"].isin([0, 4])]
    
    rng = np.random.default_rng(42)
    idx = rng.choice(len(phys_df), size=min(N_PERTURB_SAMPLES, len(phys_df)), replace=False)
    X_sub_raw = phys_df[["x0", "vx0"]].values[idx]
    X_sub = scaler.transform(X_sub_raw)
    
    f_eps_flip = np.zeros(len(EPSILONS))
    f_eps_dP = np.zeros(len(EPSILONS))
    f_eps_dU = np.zeros(len(EPSILONS))
    
    X_tensor = torch.tensor(X_sub, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        base_probs = np.stack([F.softmax(m(X_tensor), dim=1).cpu().numpy() for m in models], axis=1)
    
    base_mean = base_probs.mean(axis=1)
    base_pred = base_mean.argmax(axis=1)
    
    H_mean = -(base_mean * np.log(base_mean + 1e-12)).sum(axis=1)
    H_ind = -(base_probs * np.log(base_probs + 1e-12)).sum(axis=2)
    base_epi = H_mean - H_ind.mean(axis=1)

    thetas = rng.uniform(0, 2*np.pi, size=(len(EPSILONS), len(X_sub)))
    
    for e_idx, eps in enumerate(EPSILONS):
        perturbed_raw = np.zeros_like(X_sub_raw)
        perturbed_raw[:, 0] = X_sub_raw[:, 0] + eps * np.cos(thetas[e_idx])
        perturbed_raw[:, 1] = X_sub_raw[:, 1] + eps * np.sin(thetas[e_idx])
        perturbed_std = scaler.transform(perturbed_raw)
        
        P_tensor = torch.tensor(perturbed_std, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            p_probs = np.stack([F.softmax(m(P_tensor), dim=1).cpu().numpy() for m in models], axis=1)
            
        p_mean = p_probs.mean(axis=1)
        p_pred = p_mean.argmax(axis=1)
        
        H_p_mean = -(p_mean * np.log(p_mean + 1e-12)).sum(axis=1)
        H_p_ind = -(p_probs * np.log(p_probs + 1e-12)).sum(axis=2)
        p_epi = H_p_mean - H_p_ind.mean(axis=1)
        
        f_eps_flip[e_idx] = (p_pred != base_pred).mean()
        f_eps_dP[e_idx] = np.abs(p_mean - base_mean).sum(axis=1).mean()
        f_eps_dU[e_idx] = np.abs(p_epi - base_epi).mean()

    def get_slope(y):
        valid = y > 1e-8
        if valid.sum() < 3: return np.nan
        log_eps = np.log(EPSILONS[valid])
        log_y = np.log(y[valid])
        res = np.polyfit(log_eps, log_y, 1)
        return res[0]

    return get_slope(f_eps_flip), get_slope(f_eps_dP), get_slope(f_eps_dU)

def full_grid_box_counting(models, scaler):
    nx, nvx = 700, 700
    x_unique = np.linspace(-1.5, 1.5, nx)
    vx_unique = np.linspace(-3.0, 3.0, nvx)
    
    xx, yy = np.meshgrid(x_unique, vx_unique)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    X_full = scaler.transform(grid_points)
    X_tensor = torch.tensor(X_full, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        probs = np.stack([F.softmax(m(X_tensor), dim=1).cpu().numpy() for m in models], axis=1)
    mean_probs = probs.mean(axis=1)
    preds = mean_probs.argmax(axis=1)
    
    grid_preds = preds.reshape(nvx, nx)
    
    full_df = pd.read_csv(DATA_PATH)
    valid_mask = (~full_df["label_id"].isin([0, 4])).values.reshape(nvx, nx)
    
    boundary = np.zeros((nvx, nx), dtype=bool)
    
    same = valid_mask[:, :-1] & valid_mask[:, 1:]
    boundary[:, :-1] |= same & (grid_preds[:, :-1] != grid_preds[:, 1:])
    
    same = valid_mask[:-1, :] & valid_mask[1:, :]
    boundary[:-1, :] |= same & (grid_preds[:-1, :] != grid_preds[1:, :])
    
    boxes = [1, 2, 4, 5, 7, 10, 14, 20, 25, 28, 35, 50, 70, 100, 140, 175, 350]
    
    counts = []
    valid_boxes = []
    for b in boxes:
        b_map = boundary.reshape(nvx//b, b, nx//b, b).any(axis=(1, 3))
        c = b_map.sum()
        if c > 0:
            counts.append(np.log(c))
            valid_boxes.append(np.log(700.0/b))
            
    valid_idx = [i for i, b in enumerate(boxes) if 2 <= b <= 140]
    if len(valid_idx) < 3: return np.nan
    
    fit_b = [valid_boxes[i] for i in valid_idx]
    fit_c = [counts[i] for i in valid_idx]
    
    res = np.polyfit(fit_b, fit_c, 1)
    return res[0]

header = ["ff_scale", "hidden", "seed", "val_ba", "beta_flip", "beta_dP", "beta_dU", "D_box"]
with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"Starting sweep on {DEVICE}... Total configs: {len(FF_SCALES)*len(HIDDEN_DIMS)}")
t_start = time.time()

for ff_scale in FF_SCALES:
    for hidden in HIDDEN_DIMS:
        print(f"--- Config: ff_scale={ff_scale}, hidden={hidden} ---")
        models = []
        val_bas = []
        
        for seed in TRAIN_SEEDS:
            print(f"  Training seed {seed}...", end=" ", flush=True)
            model = BasinClassifier(hidden=hidden, ff_scale=ff_scale, ff_seed=FF_SEED).to(DEVICE)
            model, v_ba = train_model(model, seed)
            models.append(model)
            val_bas.append(v_ba)
            print(f"Done. Val BA: {v_ba:.4f}")
            
        print("  Evaluating Beta...", end=" ", flush=True)
        b_flip, b_dP, b_dU = compute_beta_metrics(models, scaler)
        print(f"Done. Flip: {b_flip:.3f}, dP: {b_dP:.3f}, dU: {b_dU:.3f}")
        
        print("  Computing Box-Counting Dimension...", end=" ", flush=True)
        d_box = full_grid_box_counting(models, scaler)
        print(f"Done. D_box: {d_box:.3f}")
        
        for seed, v_ba in zip(TRAIN_SEEDS, val_bas):
            row = [ff_scale, hidden, seed, v_ba, b_flip, b_dP, b_dU, d_box]
            with open(SUMMARY_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

print(f"\nSweep complete in {(time.time()-t_start)/60:.1f} minutes.")
print(f"Results saved to {SUMMARY_CSV}")