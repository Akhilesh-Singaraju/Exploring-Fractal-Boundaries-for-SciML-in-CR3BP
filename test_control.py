import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = "files/synthetic_results"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "control.csv")

FF_SCALE = 5.0
HIDDEN = 512
FF_SEED = 0
TRAIN_SEEDS = [42, 123, 456, 789, 1011]

EPOCHS = 80
BATCH_SIZE = 1024
LR = 3e-4
RES = 512

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
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SyntheticClassifier(nn.Module):
    def __init__(
        self,
        input_dim=2,
        num_classes=3,
        hidden=512,
        n_blocks=3,
        ff_dim=256,
        ff_scale=5.0,
        dropout=0.1,
        ff_seed=0,
    ):
        super().__init__()

        self.ff = FourierFeatures(input_dim, ff_dim, ff_scale, ff_seed)

        in_dim = 2 * ff_dim + input_dim

        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(hidden, dropout) for _ in range(n_blocks)]
        )

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(self.ff(x))))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma

        if alpha is not None:
            self.register_buffer(
                "alpha",
                torch.tensor(alpha, dtype=torch.float32)
            )
        else:
            self.alpha = None

    def forward(self, logits, target):
        log_pt = -F.cross_entropy(
            logits,
            target,
            reduction="none"
        )

        pt = torch.exp(log_pt)

        focal_weight = (
            1.0 - pt
        ) ** self.gamma

        loss = focal_weight * (-log_pt)

        if self.alpha is not None:
            alpha = self.alpha.to(
                device=target.device,
                dtype=logits.dtype
            )

            loss = alpha[target] * loss

        return loss.mean()


def generate_voronoi(res=512):
    points = np.array(
        [[0.0, 1.0], [-np.sqrt(3) / 2.0, -0.5], [np.sqrt(3) / 2.0, -0.5]], dtype=float
    )

    x = np.linspace(-2.0, 2.0, res)

    y = np.linspace(-2.0, 2.0, res)

    xx, yy = np.meshgrid(x, y)

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    dists = np.linalg.norm(grid_points[:, None, :] - points[None, :, :], axis=2)

    labels = np.argmin(dists, axis=1)

    return (xx.ravel(), yy.ravel(), labels, xx, yy)


def generate_newton(res=512, max_iter=100, tolerance=1e-5):
    x = np.linspace(-2.0, 2.0, res)

    y = np.linspace(-2.0, 2.0, res)

    xx, yy = np.meshgrid(x, y)

    z = xx.astype(np.complex128) + 1j * yy.astype(np.complex128)

    roots = np.array(
        [1.0 + 0.0j, -0.5 + 0.8660254037844386j, -0.5 - 0.8660254037844386j],
        dtype=np.complex128,
    )

    for _ in range(max_iter):
        z2 = z * z
        denominator = 3.0 * z2
        numerator = z2 * z - 1.0

        safe = np.abs(denominator) > 1e-14

        z_new = z.copy()

        z_new[safe] = z[safe] - numerator[safe] / denominator[safe]

        z = z_new

    distances = np.stack([np.abs(z - root) for root in roots], axis=-1)

    labels = np.argmin(distances, axis=-1)

    converged = np.min(distances, axis=-1) < tolerance

    labels = labels.astype(np.int64)
    labels[~converged] = -1

    return (xx.ravel(), yy.ravel(), labels.ravel(), xx, yy)


def compute_box_counting_2d(grid_labels, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.ones_like(grid_labels, dtype=bool)

    boundary = np.zeros_like(valid_mask, dtype=bool)

    horizontal = (
        valid_mask[:, :-1]
        & valid_mask[:, 1:]
        & (grid_labels[:, :-1] != grid_labels[:, 1:])
    )

    vertical = (
        valid_mask[:-1, :]
        & valid_mask[1:, :]
        & (grid_labels[:-1, :] != grid_labels[1:, :])
    )

    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical

    boxes = [1, 2, 4, 8, 16, 32, 64, 128]

    counts = []
    scales = []

    for b in boxes:
        if grid_labels.shape[0] % b != 0 or grid_labels.shape[1] % b != 0:
            continue

        b_map = boundary.reshape(
            grid_labels.shape[0] // b, b, grid_labels.shape[1] // b, b
        ).any(axis=(1, 3))

        count = b_map.sum()

        if count > 0:
            counts.append(np.log(count))

            scales.append(np.log(grid_labels.shape[0] / b))

    if len(scales) < 3:
        return np.nan

    scales = np.asarray(scales)
    counts = np.asarray(counts)

    return np.polyfit(scales, counts, 1)[0]


def get_slope(y):
    y = np.asarray(y)

    valid = np.isfinite(y) & (y > 1e-8)

    if valid.sum() < 3:
        return np.nan

    return np.polyfit(np.log(EPSILONS[valid]), np.log(y[valid]), 1)[0]


def ensemble_probabilities(models, X_tensor):
    with torch.no_grad():
        probabilities = np.stack(
            [F.softmax(model(X_tensor), dim=1).cpu().numpy() for model in models],
            axis=1,
        )

    return probabilities


def evaluate_perturbations(models, X_sub, X_sub_raw, scaler, rng):
    X_tensor = torch.tensor(X_sub, dtype=torch.float32).to(DEVICE)

    base_probs = ensemble_probabilities(models, X_tensor)

    base_mean = base_probs.mean(axis=1)

    base_pred = base_mean.argmax(axis=1)

    H_mean = -(base_mean * np.log(base_mean + 1e-12)).sum(axis=1)

    H_ind = -(base_probs * np.log(base_probs + 1e-12)).sum(axis=2)

    base_epi = H_mean - H_ind.mean(axis=1)

    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(len(EPSILONS), len(X_sub)))

    f_eps_flip = np.zeros(len(EPSILONS))

    f_eps_dP = np.zeros(len(EPSILONS))

    f_eps_dU = np.zeros(len(EPSILONS))

    for e_idx, eps in enumerate(EPSILONS):
        perturbed_raw = np.zeros_like(X_sub_raw)

        perturbed_raw[:, 0] = X_sub_raw[:, 0] + eps * np.cos(thetas[e_idx])

        perturbed_raw[:, 1] = X_sub_raw[:, 1] + eps * np.sin(thetas[e_idx])

        perturbed_std = scaler.transform(perturbed_raw)

        P_tensor = torch.tensor(perturbed_std, dtype=torch.float32).to(DEVICE)

        p_probs = ensemble_probabilities(models, P_tensor)

        p_mean = p_probs.mean(axis=1)

        p_pred = p_mean.argmax(axis=1)

        H_p_mean = -(p_mean * np.log(p_mean + 1e-12)).sum(axis=1)

        H_p_ind = -(p_probs * np.log(p_probs + 1e-12)).sum(axis=2)

        p_epi = H_p_mean - H_p_ind.mean(axis=1)

        f_eps_flip[e_idx] = (p_pred != base_pred).mean()

        f_eps_dP[e_idx] = np.abs(p_mean - base_mean).sum(axis=1).mean()

        f_eps_dU[e_idx] = np.abs(p_epi - base_epi).mean()

    return (get_slope(f_eps_flip), get_slope(f_eps_dP), get_slope(f_eps_dU))


def run_control(x, y, labels_1d, xx, yy, name):
    print(f"\n=== {name} ===")

    full_labels = labels_1d.reshape(RES, RES)

    full_valid = full_labels != -1

    valid = labels_1d != -1

    df = pd.DataFrame({"x0": x[valid], "vx0": y[valid], "label_id": labels_1d[valid]})

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )

    X_train = train_df[["x0", "vx0"]].values

    y_train = train_df["label_id"].values

    X_test = test_df[["x0", "vx0"]].values

    y_test = test_df["label_id"].values

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)

    X_test_std = scaler.transform(X_test)

    class_counts = np.bincount(y_train, minlength=3)

    safe_counts = np.maximum(class_counts, 1)

    raw_weights = len(y_train) / (3.0 * safe_counts)

    class_weights = raw_weights / np.mean(raw_weights)

    train_dataset = data.TensorDataset(
        torch.tensor(X_train_std, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    test_dataset = data.TensorDataset(
        torch.tensor(X_test_std, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    models = []
    test_bas = []

    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
        )

        test_loader = data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        model = SyntheticClassifier(
            hidden=HIDDEN, ff_scale=FF_SCALE, ff_seed=FF_SEED
        ).to(DEVICE)

        criterion = FocalLoss(alpha=class_weights, gamma=1.0)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        for _ in range(EPOCHS):
            model.train()

            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            scheduler.step()

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)

                preds = model(inputs).argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())

                all_targets.extend(labels.numpy())

        ba = balanced_accuracy_score(all_targets, all_preds)

        test_bas.append(ba)
        models.append(model)

        print(f"  Seed {seed}: BA={ba:.4f}")

    rng = np.random.default_rng(42)

    n_samples = min(N_PERTURB_SAMPLES, len(X_test_std))

    idx = rng.choice(len(X_test_std), size=n_samples, replace=False)

    X_sub = X_test_std[idx]

    X_sub_raw = scaler.inverse_transform(X_sub)

    b_flip, b_dP, b_dU = evaluate_perturbations(models, X_sub, X_sub_raw, scaler, rng)

    X_full = np.column_stack([xx.ravel(), yy.ravel()])

    X_full_std = scaler.transform(X_full)

    X_tensor = torch.tensor(X_full_std, dtype=torch.float32).to(DEVICE)

    probs = ensemble_probabilities(models, X_tensor)

    learned_preds = probs.mean(axis=1).argmax(axis=1).reshape(RES, RES)

    d_box_learned = compute_box_counting_2d(learned_preds)

    d_box_true = compute_box_counting_2d(full_labels, valid_mask=full_valid)

    d_box_error = d_box_learned - d_box_true

    print(f"  True D_box: {d_box_true:.4f}")

    print(f"  Learned D_box: {d_box_learned:.4f}")

    print(f"  D_box error: {d_box_error:.4f}")

    print(f"  Beta flip: {b_flip:.4f}")

    print(f"  Beta dP: {b_dP:.4f}")

    print(f"  Beta dU: {b_dU:.4f}")

    print(f"  Mean BA: {np.mean(test_bas):.4f}")

    return (
        np.mean(test_bas),
        d_box_true,
        d_box_learned,
        d_box_error,
        b_flip,
        b_dP,
        b_dU,
    )


header = [
    "dataset",
    "val_ba",
    "D_box_true",
    "D_box_learned",
    "D_box_error",
    "beta_flip",
    "beta_dP",
    "beta_dU",
]

with open(SUMMARY_CSV, "w", newline="") as f:
    csv.writer(f).writerow(header)


x, y, labels, xx, yy = generate_voronoi(RES)

result = run_control(x, y, labels, xx, yy, "Voronoi Smooth")

with open(SUMMARY_CSV, "a", newline="") as f:
    csv.writer(f).writerow(["Voronoi"] + list(result))


x, y, labels, xx, yy = generate_newton(RES)

n_invalid = np.sum(labels == -1)

print(
    f"\nNewton non-converged points: "
    f"{n_invalid}/{labels.size} "
    f"({100.0 * n_invalid / labels.size:.6f}%)"
)

result = run_control(x, y, labels, xx, yy, "Newton Fractal")

with open(SUMMARY_CSV, "a", newline="") as f:
    csv.writer(f).writerow(["Newton"] + list(result))


print(f"\nDone. Saved to {SUMMARY_CSV}")
