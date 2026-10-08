import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
C_VALUES = [3.00, 3.05, 3.10, 3.14]
GRID_SIZES = [100, 300, 500, 700]
EVAL_GRID_SIZE = 700

STRICT_COMMON_ACROSS_C = False

EPOCHS = 150
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
PATIENCE = 30

EVAL_BINS = 20
EVAL_FRAC = 0.15
VAL_FRAC = 0.15
SPLIT_SEED = 42

X_MIN, X_MAX = -1.5, 1.5
VX_MIN, VX_MAX = -3.0, 3.0

LABEL_CLASSES = np.array([1, 2, 3])


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
        log_pt = -F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(log_pt)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * (-log_pt)

        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            loss = self.alpha[target] * loss

        return loss.mean()


def get_block_edges():
    x_edges = np.linspace(X_MIN, X_MAX, EVAL_BINS + 1)
    vx_edges = np.linspace(VX_MIN, VX_MAX, EVAL_BINS + 1)
    return x_edges, vx_edges


def assign_block_ids(df):
    x_edges, vx_edges = get_block_edges()

    xb = np.digitize(df["x0"].values, x_edges) - 1
    vxb = np.digitize(df["vx0"].values, vx_edges) - 1

    xb = np.clip(xb, 0, EVAL_BINS - 1)
    vxb = np.clip(vxb, 0, EVAL_BINS - 1)

    return xb * EVAL_BINS + vxb


def get_fixed_block_split():
    rng = np.random.RandomState(SPLIT_SEED)

    all_blocks = np.arange(EVAL_BINS * EVAL_BINS)

    n_eval = max(1, int(len(all_blocks) * EVAL_FRAC))
    eval_blocks = rng.choice(all_blocks, size=n_eval, replace=False)

    remaining = np.setdiff1d(all_blocks, eval_blocks)
    n_val = max(1, int(len(remaining) * (VAL_FRAC / (1.0 - EVAL_FRAC))))
    val_blocks = rng.choice(remaining, size=n_val, replace=False)

    return set(eval_blocks.tolist()), set(val_blocks.tolist())


EVAL_BLOCKS, VAL_BLOCKS = get_fixed_block_split()


def train_single_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       epochs, patience, device, model_idx, c_val, res):
    best_val_ba = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

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
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_ba": val_ba})

        if val_ba > best_val_ba + 1e-4:
            best_val_ba = val_ba
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [C={c_val:.2f} {res}x{res} Model {model_idx+1}] Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | Val BalAcc: {val_ba:.4f}")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (Best Val BalAcc: {best_val_ba:.4f})")
            break

    model.load_state_dict(best_state)
    return model, best_val_ba, history


def evaluate_ensemble(models, test_loader, label_encoder, device):
    all_preds = []
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

            batch_probs = np.stack(batch_probs, axis=1)
            mean_probs = batch_probs.mean(axis=1)
            preds = mean_probs.argmax(axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    ba = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print("\n  Ensemble Classification Report:")
    print(classification_report(
        all_targets, all_preds,
        target_names=[f"Class {c}" for c in label_encoder.classes_],
        zero_division=0
    ))

    print("  Ensemble Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print(f"  Ensemble Balanced Accuracy: {ba:.4f}")
    print(f"  Ensemble Macro F1: {macro_f1:.4f}")

    return ba


def save_training_curves(all_histories, C):
    fig_dir = "files/figures"
    pkl_dir = "files/pickles"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)

    if not all_histories:
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    palette = sns.color_palette("tab10", n_colors=len(all_histories))

    for (grid_label, histories), color in zip(all_histories.items(), palette):
        max_epoch = max(len(h) for h in histories)
        mean_ba = np.zeros(max_epoch)
        mean_loss = np.zeros(max_epoch)
        counts = np.zeros(max_epoch)

        for h in histories:
            for entry in h:
                idx = entry["epoch"] - 1
                mean_ba[idx] += entry["val_ba"]
                mean_loss[idx] += entry["train_loss"]
                counts[idx] += 1

        valid = counts > 0
        epochs_arr = np.arange(1, max_epoch + 1)[valid]
        mean_ba = (mean_ba / counts)[valid]
        mean_loss = (mean_loss / counts)[valid]

        axes[0].plot(epochs_arr, mean_loss, color=color, linewidth=1.8, label=grid_label)
        axes[1].plot(epochs_arr, mean_ba, color=color, linewidth=1.8, label=grid_label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title(f"Training Loss, C={C:.2f}")
    axes[0].legend(loc="upper right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Balanced Accuracy")
    axes[1].set_title(f"Validation Balanced Accuracy, C={C:.2f}")
    axes[1].legend(loc="lower right")

    fig.tight_layout()

    png_path = os.path.join(fig_dir, f"training_curves_C{C:.2f}.png")
    pkl_path = os.path.join(pkl_dir, f"training_curves_C{C:.2f}.pkl")

    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    with open(pkl_path, "wb") as f:
        pickle.dump(fig, f)

    plt.close(fig)

    print(f"\n  Saved training curves: {png_path}")
    print(f"  Saved pickle: {pkl_path}")


base_dfs = {}
physical_masks = []
valid_C = []

for C in C_VALUES:
    base_path = f"files/data/{C:.2f}/{EVAL_GRID_SIZE}x{EVAL_GRID_SIZE}.csv"

    if not os.path.exists(base_path):
        print(f"Missing base evaluation grid: {base_path}")
        continue

    df = pd.read_csv(base_path)
    df["block_id"] = assign_block_ids(df)

    base_dfs[C] = df
    physical_masks.append(~df["label_id"].isin([0, 4]).values)
    valid_C.append(C)

if len(valid_C) == 0:
    raise RuntimeError("No valid 700x700 base grids found.")

strict_common = STRICT_COMMON_ACROSS_C

if strict_common:
    lengths = [len(df) for df in base_dfs.values()]
    if len(set(lengths)) != 1:
        print("Base grids have different lengths. Disabling strict common evaluation.")
        strict_common = False
    else:
        common_physical = np.stack(physical_masks).all(axis=0)

for C in valid_C:
    print(f"\n{'='*60}")
    print(f"C = {C:.2f}")
    print(f"{'='*60}")

    base_df = base_dfs[C]
    base_physical = (~base_df["label_id"].isin([0, 4])).values

    if strict_common:
        common_mask = common_physical
    else:
        common_mask = base_physical

    eval_mask = common_mask & base_df["block_id"].isin(EVAL_BLOCKS).values
    val_mask = common_mask & base_df["block_id"].isin(VAL_BLOCKS).values

    common_eval_df = base_df[eval_mask].copy()
    common_val_df = base_df[val_mask].copy()

    print(f"Common eval rows: {len(common_eval_df)}")
    print(f"Common val rows:  {len(common_val_df)}")

    if len(common_eval_df) == 0 or len(common_val_df) == 0:
        print("Empty common evaluation or validation set, skipping this C.")
        continue

    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_CLASSES)
    num_classes = len(label_encoder.classes_)

    X_val_raw = common_val_df[["x0", "vx0"]].values
    y_val = label_encoder.transform(common_val_df["label_id"].values)

    X_test_raw = common_eval_df[["x0", "vx0"]].values
    y_test = label_encoder.transform(common_eval_df["label_id"].values)

    all_histories = {}

    for res in GRID_SIZES:
        print(f"\n{'-'*60}")
        print(f"Training Ensemble for C = {C:.2f}, Grid = {res}x{res}")
        print(f"{'-'*60}")

        if res == EVAL_GRID_SIZE:
            train_df = base_df.copy()
        else:
            csv_path = f"files/data/{C:.2f}/{res}x{res}.csv"
            if not os.path.exists(csv_path):
                print(f"Missing {csv_path}, skipping.")
                continue

            train_df = pd.read_csv(csv_path)
            train_df["block_id"] = assign_block_ids(train_df)

        train_df = train_df[~train_df["label_id"].isin([0, 4])].copy()
        train_df = train_df[~train_df["block_id"].isin(EVAL_BLOCKS)].copy()
        train_df = train_df[~train_df["block_id"].isin(VAL_BLOCKS)].copy()

        if len(train_df) == 0:
            print("Empty training set after excluding eval/val blocks, skipping.")
            continue

        X_train_raw = train_df[["x0", "vx0"]].values
        y_train_raw = train_df["label_id"].values
        y_train = label_encoder.transform(y_train_raw)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        print(f"  Train rows: {len(X_train)}")
        print(f"  Val rows:   {len(X_val)}")
        print(f"  Test rows:  {len(X_test)}")

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
        safe_counts = np.maximum(class_counts, 1)
        raw_weights = len(y_train) / (num_classes * safe_counts)
        raw_weights = np.clip(raw_weights, 0.0, 5.0)
        class_weights = raw_weights / np.mean(raw_weights)

        print(f"  Class weights: {class_weights}")

        criterion = FocalLoss(alpha=class_weights, gamma=1.0)

        models = []
        grid_histories = []
        save_dir = f"files/models/{C:.2f}/{res}x{res}"
        os.makedirs(save_dir, exist_ok=True)

        split_seed = SPLIT_SEED + int(C * 100) + res

        for i in range(num_ensembles):
            print(f"\n  --- Training Model {i+1}/{num_ensembles} ---")
            model_seed = split_seed + i + 1
            torch.manual_seed(model_seed)

            model = BasinClassifier(
                input_dim=2,
                num_classes=num_classes,
                hidden=256,
                n_blocks=3,
                ff_dim=256,
                ff_scale=5.0,
                ff_seed=model_seed
            ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            model, best_ba, history = train_single_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                EPOCHS,
                PATIENCE,
                device,
                i,
                C,
                res
            )

            models.append(model)
            grid_histories.append(history)

            torch.save(model.state_dict(), os.path.join(save_dir, f"MODEL_{i}.pt"))

        torch.save({
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "label_classes": label_encoder.classes_,
            "eval_blocks": sorted(EVAL_BLOCKS),
            "val_blocks": sorted(VAL_BLOCKS),
            "strict_common_across_C": strict_common,
        }, os.path.join(save_dir, "preprocessing.pt"))

        print(f"\n  --- Evaluating Ensemble on Common Test Set ---")
        evaluate_ensemble(models, test_loader, label_encoder, device)

        all_histories[f"{res}x{res}"] = grid_histories

    save_training_curves(all_histories, C)