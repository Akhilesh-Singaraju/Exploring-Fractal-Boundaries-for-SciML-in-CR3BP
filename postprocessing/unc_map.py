import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_ENSEMBLE = 5
C_VALUES = [3.00, 3.05, 3.10, 3.14]
GRID_SIZES = [100, 300, 500, 700]
DATA_DIR = "files/data"
MODEL_DIR = "files/models"
PLOT_DIR = "files/figures"
PICKLE_DIR = "files/pickles"


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


def load_ensemble(C, res):
    save_dir = os.path.join(MODEL_DIR, f"{C:.2f}", f"{res}x{res}")
    pre = torch.load(os.path.join(save_dir, "preprocessing.pt"), map_location="cpu", weights_only=False)
    mean, scale = pre["scaler_mean"], pre["scaler_scale"]
    classes = pre["label_classes"]
    split_seed = 42 + int(C * 100) + res

    models = []
    for i in range(NUM_ENSEMBLE):
        m = BasinClassifier(
            input_dim=2, num_classes=len(classes), hidden=256, n_blocks=3,
            ff_dim=256, ff_scale=5.0, ff_seed=split_seed + i + 1,
        )
        sd = torch.load(os.path.join(save_dir, f"MODEL_{i}.pt"), map_location="cpu", weights_only=False)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, mean, scale, classes


def compute_grid_uncertainty(C, res, batch_size=8192):
    models, mean, scale, classes = load_ensemble(C, res)
    models = [m.to(DEVICE) for m in models]

    csv_path = os.path.join(DATA_DIR, f"{C:.2f}", f"{res}x{res}.csv")
    df = pd.read_csv(csv_path)
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

    mean_probs = all_probs.mean(axis=1)
    mean_probs = np.clip(mean_probs, 1e-12, 1.0)
    all_probs = np.clip(all_probs, 1e-12, 1.0)

    H_mean = -(mean_probs * np.log(mean_probs)).sum(axis=1)
    H_ind = -(all_probs * np.log(all_probs)).sum(axis=2)
    epi = H_mean - H_ind.mean(axis=1)

    phys["prob_earth"] = mean_probs[:, 0]
    phys["prob_moon"] = mean_probs[:, 1]
    phys["prob_escape"] = mean_probs[:, 2]
    phys["pred_label_id"] = classes[mean_probs.argmax(axis=1)]
    phys["predictive_entropy"] = H_mean
    phys["epi_uncertainty"] = epi

    out_dir = os.path.join(DATA_DIR, f"{C:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"uncertainty_{res}x{res}.csv")
    phys.to_csv(out, index=False)
    print(f"  Saved {out} | N={N} | mean U_epi={epi.mean():.4f} | max U_epi={epi.max():.4f}")
    return phys, df


def plot_uncertainty_maps(C, res, phys, full):
    x_unique = np.sort(full["x0"].unique())
    vx_unique = np.sort(full["vx0"].unique())
    nx, nvx = len(x_unique), len(vx_unique)

    Z_lab = full["label_id"].values.reshape(nvx, nx)
    valid = np.isin(Z_lab, [1, 2, 3])

    g_true = np.where(valid, Z_lab, np.nan)
    g_pred = np.full(Z_lab.shape, np.nan)
    g_epi = np.full(Z_lab.shape, np.nan)
    g_pred[valid] = phys["pred_label_id"].values
    g_epi[valid] = phys["epi_uncertainty"].values

    extent = [x_unique[0], x_unique[-1], vx_unique[0], vx_unique[-1]]

    cmap3 = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
    cmap3.set_bad("lightgray")
    hot = plt.get_cmap("hot").copy()
    hot.set_bad("lightgray")

    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=200)

    axes[0].imshow(g_true, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap3, vmin=1, vmax=3, interpolation="nearest")
    axes[0].set_title(f"True DOP853 Basins\nC={C:.2f}, {res}×{res}", fontsize=12)
    axes[0].set_xlabel(r"$x_0$ [LU]")
    axes[0].set_ylabel(r"$vx_0$ [VU]")

    axes[1].imshow(g_pred, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap3, vmin=1, vmax=3, interpolation="nearest")
    axes[1].set_title("Ensemble Prediction\n(Mean Softmax)", fontsize=12)
    axes[1].set_xlabel(r"$x_0$ [LU]")

    epi_valid = g_epi[~np.isnan(g_epi)]
    vmax_epi = np.percentile(epi_valid, 99) if len(epi_valid) > 0 else 1.0

    im = axes[2].imshow(g_epi, origin="lower", extent=extent, aspect="auto",
                        cmap=hot, vmin=0, vmax=vmax_epi,
                        interpolation="nearest")
    axes[2].set_title("Epistemic Uncertainty\n(Mutual Information)", fontsize=12)
    axes[2].set_xlabel(r"$x_0$ [LU]")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()

    png_path = os.path.join(PLOT_DIR, f"uncertainty_C{C:.2f}_{res}x{res}.png")
    pkl_path = os.path.join(PICKLE_DIR, f"uncertainty_C{C:.2f}_{res}x{res}.pkl")

    fig.savefig(png_path, bbox_inches="tight", dpi=200)

    with open(pkl_path, "wb") as f:
        pickle.dump(fig, f)

    plt.close(fig)
    print(f"  Saved {png_path}")
    print(f"  Saved {pkl_path}")


if __name__ == "__main__":
    for C in C_VALUES:
        print(f"\n{'='*60}")
        print(f"C = {C:.2f}")
        print(f"{'='*60}")

        for res in GRID_SIZES:
            print(f"\n--- {res}×{res} ---")
            try:
                phys, full = compute_grid_uncertainty(C, res)
                plot_uncertainty_maps(C, res, phys, full)
            except FileNotFoundError as e:
                print(f"  Missing files for C={C:.2f} res={res}, skipping: {e}")
            except Exception as e:
                print(f"  Error processing C={C:.2f} res={res}: {e}")