"""
Full-grid ensemble epistemic uncertainty maps + visualization.
Produces files/data/uncertainty_C{C}.csv and files/plots/uncertainty_C{C}.png
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")
NUM_ENSEMBLE = 5
C_VALUES = [3.00, 3.05, 3.10, 3.14]
DATA_DIR = "files/data"
MODEL_DIR = "files/models"
PLOT_DIR = "files/plots"

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

def load_ensemble(C):
    save_dir = os.path.join(MODEL_DIR, f"{C:.2f}")
    pre = torch.load(os.path.join(save_dir, "preprocessing.pt"), map_location="cpu")
    mean, scale = pre["scaler_mean"], pre["scaler_scale"]
    classes = pre["label_classes"]
    split_seed = 42 + int(C * 100)

    models = []
    for i in range(NUM_ENSEMBLE):
        m = BasinClassifier(
            input_dim=2, num_classes=len(classes), hidden=256, n_blocks=3,
            ff_dim=256, ff_scale=5.0, ff_seed=split_seed + i + 1,
        )
        sd = torch.load(os.path.join(save_dir, f"MODEL_{i}.pt"), map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, mean, scale, classes

def compute_full_grid_uncertainty(C, batch_size=8192):
    models, mean, scale, classes = load_ensemble(C)
    models = [m.to(DEVICE) for m in models]

    df = pd.read_csv(os.path.join(DATA_DIR, f"basin_C{C:.2f}.csv"))
    phys = df[df["label_id"].isin([1, 2, 3])].copy().reset_index(drop=True)

    X = phys[["x0", "vx0"]].to_numpy(dtype=np.float64)
    Xs = torch.tensor((X - mean) / scale, dtype=torch.float32)

    N, M, K = len(Xs), len(models), len(classes)
    all_probs = np.empty((N, M, K), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, batch_size):
            b = Xs[s:s + batch_size].to(DEVICE)
            for mi, m in enumerate(models):
                all_probs[s:s + batch_size, mi, :] = F.softmax(m(b), dim=1).cpu().numpy()

    mean_probs = all_probs.mean(axis=1)                      # (N, K)
    eps = 1e-12
    H_mean = -(mean_probs * np.log(mean_probs + eps)).sum(axis=1)
    H_ind = -(all_probs * np.log(all_probs + eps)).sum(axis=2)   # (N, M)
    epi = H_mean - H_ind.mean(axis=1)                        

    phys["prob_earth"] = mean_probs[:, 0]
    phys["prob_moon"] = mean_probs[:, 1]
    phys["prob_escape"] = mean_probs[:, 2]
    phys["pred_label_id"] = mean_probs.argmax(axis=1) + 1
    phys["predictive_entropy"] = H_mean
    phys["epi_uncertainty"] = epi

    out = os.path.join(DATA_DIR, f"uncertainty_C{C:.2f}.csv")
    phys.to_csv(out, index=False)
    print(f"Saved {out} | N={N} | mean U_epi={epi.mean():.4f} | max U_epi={epi.max():.4f}")
    return phys

def plot_uncertainty_maps(C, phys):
    os.makedirs(PLOT_DIR, exist_ok=True)
    extent = [phys["x0"].min(), phys["x0"].max(), phys["vx0"].min(), phys["vx0"].max()]
    cmap3 = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=200)

    g_true = phys.pivot(index="vx0", columns="x0", values="label_id")
    axes[0].imshow(g_true.values, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap3, vmin=1, vmax=3)
    axes[0].set_title(f"True DOP853 basins (C={C:.2f})")

    g_pred = phys.pivot(index="vx0", columns="x0", values="pred_label_id")
    axes[1].imshow(g_pred.values, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap3, vmin=1, vmax=3)
    axes[1].set_title("Ensemble prediction (mean softmax)")

    g_epi = phys.pivot(index="vx0", columns="x0", values="epi_uncertainty")
    im = axes[2].imshow(g_epi.values, origin="lower", extent=extent, aspect="auto",
                        cmap="hot", vmax=np.percentile(g_epi.values, 99))
    axes[2].set_title("Epistemic uncertainty (mutual information)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x0 [LU]")
    axes[0].set_ylabel("vx0 [VU]")

    path = os.path.join(PLOT_DIR, f"uncertainty_C{C:.2f}.png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

if __name__ == "__main__":
    for C in C_VALUES:
        phys = compute_full_grid_uncertainty(C)
        plot_uncertainty_maps(C, phys)