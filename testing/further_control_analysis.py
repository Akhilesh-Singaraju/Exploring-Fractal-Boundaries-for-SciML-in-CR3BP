from pathlib import Path
import re
import csv
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

BASE_DIR = Path("files")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "sweep_results"
C_VALUES = [3.00, 3.05, 3.10, 3.14]
RES_LIST = [100, 300, 500, 700]
MI_C_VALUE = 3.10
MI_RESOLUTION = 700
VALID_LABELS = {1, 2, 3}
PRED_LABEL_MAP = {0: 1, 1: 2, 2: 3}
PREDICTION_COLUMN = None
PROBABILITY_COLUMNS = None
MI_COLUMN = "epi_uncertainty"
ENTROPY_COLUMN = "predictive_entropy"
UNCERTAINTY_FILE_TEMPLATE = "uncertainty_{res}x{res}.csv"
RAW_FILE_TEMPLATE = "{res}x{res}.csv"
BOX_FRACTIONS = np.geomspace(0.02, 0.20, 12)
N_PERTURB = 2000
N_BOOT = 500
BLOCK_PIXELS = 10
EPSILONS = np.logspace(np.log10(0.005), np.log10(0.05), 8)
RANDOM_SEED = 42
MIN_RESPONSE = 1e-12
RUN_DBOX_SWEEP = True
RUN_BETA_BOOTSTRAP = True
RUN_MI_LOCALIZATION = True
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR.mkdir(parents=True, exist_ok=True)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_features=256, scale=5.0, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(in_dim, num_features, generator=generator) * scale
        self.register_buffer("B", matrix)

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
    def __init__(self, input_dim=2, num_classes=3, hidden=256, n_blocks=3, ff_dim=256, ff_scale=5.0, dropout=0.1, ff_seed=0):
        super().__init__()
        self.ff = FourierFeatures(input_dim, ff_dim, ff_scale, ff_seed)
        in_dim = 2 * ff_dim + input_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(self.ff(x))))


def load_frame(c_value, res, require_uncertainty=False):
    folder = DATA_DIR / f"{c_value:.2f}"
    uncertainty_path = folder / UNCERTAINTY_FILE_TEMPLATE.format(res=res)
    raw_path = folder / RAW_FILE_TEMPLATE.format(res=res)
    selected_path = uncertainty_path if uncertainty_path.exists() else raw_path
    if not selected_path.exists():
        kind = "uncertainty" if require_uncertainty else "data"
        raise FileNotFoundError(f"Missing {kind} CSV for C={c_value:.2f}, resolution={res}: expected {uncertainty_path} or {raw_path}")
    frame = pd.read_csv(selected_path)
    if "x0" not in frame or "vx0" not in frame:
        raise ValueError(f"{selected_path} must contain x0 and vx0 columns")
    if "label_id" not in frame:
        if not raw_path.exists():
            raise ValueError(f"{selected_path} has no label_id column and no raw data file is available at {raw_path}")
        raw = pd.read_csv(raw_path)
        if not {"x0", "vx0", "label_id"}.issubset(raw.columns):
            raise ValueError(f"{raw_path} must contain x0, vx0, and label_id")
        frame = frame.copy()
        raw = raw[["x0", "vx0", "label_id"]].copy()
        frame["_xkey"] = np.round(pd.to_numeric(frame["x0"], errors="raise"), 10)
        frame["_vkey"] = np.round(pd.to_numeric(frame["vx0"], errors="raise"), 10)
        raw["_xkey"] = np.round(pd.to_numeric(raw["x0"], errors="raise"), 10)
        raw["_vkey"] = np.round(pd.to_numeric(raw["vx0"], errors="raise"), 10)
        frame = frame.merge(raw[["_xkey", "_vkey", "label_id"]], on=["_xkey", "_vkey"], how="left", validate="one_to_one")
        frame = frame.drop(columns=["_xkey", "_vkey"])
        if frame["label_id"].isna().any():
            raise ValueError(f"Could not join every row in {selected_path} to label_id in {raw_path}; check coordinate precision")
    if require_uncertainty:
        missing = [name for name in (MI_COLUMN, ENTROPY_COLUMN) if name not in frame.columns]
        if missing:
            raise ValueError(f"{selected_path} is missing required uncertainty columns {missing}")
    if frame[["x0", "vx0"]].isna().any().any():
        raise ValueError(f"Non-finite grid coordinates in {selected_path}")
    frame = frame.copy().reset_index(drop=True)
    return frame, selected_path


def validated_grid_indices(frame, res, x_bounds=(-1.5, 1.5), vx_bounds=(-3.0, 3.0), allow_missing=True):
    """
    Map rows onto the *known simulation grid*, rather than inferring the grid
    from unique coordinates. This supports files where rows/classes were filtered.
    Missing grid cells remain missing; duplicate mapped cells are still rejected.
    """
    x_grid = np.linspace(x_bounds[0], x_bounds[1], res)
    vx_grid = np.linspace(vx_bounds[0], vx_bounds[1], res)

    def map_axis(values, grid, axis_name):
        vals = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
        idx = np.searchsorted(grid, vals)
        idx = np.clip(idx, 0, len(grid) - 1)
        left = np.maximum(idx - 1, 0)
        use_left = np.abs(grid[left] - vals) < np.abs(grid[idx] - vals)
        idx[use_left] = left[use_left]
        spacing = float(np.min(np.diff(grid)))
        tolerance = max(1e-9, spacing * 1e-3)
        delta = np.abs(grid[idx] - vals)
        if np.any(delta > tolerance):
            bad = float(np.max(delta))
            raise ValueError(
                f"{axis_name} coordinates do not match the expected simulation grid "
                f"with bounds {grid[0]:g}..{grid[-1]:g}; max mismatch={bad:g}. "
                "Confirm the original grid bounds/resolution."
            )
        return idx.astype(int)

    ix = map_axis(frame["x0"].to_numpy(), x_grid, "x0")
    iy = map_axis(frame["vx0"].to_numpy(), vx_grid, "vx0")
    linear = iy * res + ix
    if len(np.unique(linear)) != len(linear):
        raise ValueError("Duplicate (x0, vx0) rows map to the same expected grid cell.")
    n_missing = res * res - len(np.unique(linear))
    if n_missing and not allow_missing:
        raise ValueError(f"Expected a complete {res}x{res} grid; {n_missing} cells are missing.")
    if n_missing:
        warnings.warn(
            f"Input has {len(frame)} rows mapped to the expected {res}x{res} grid; "
            f"{n_missing} cells are absent. Geometry will be masked at absent cells.",
            RuntimeWarning,
        )
    return x_grid, vx_grid, ix, iy


def resolve_prediction_column(frame):
    if PREDICTION_COLUMN is not None:
        if PREDICTION_COLUMN not in frame.columns:
            raise ValueError(f"Configured prediction column {PREDICTION_COLUMN!r} is absent")
        return PREDICTION_COLUMN
    candidates = [
        "pred_label", "predicted_label", "predicted_label_id", "pred_id", "pred_class", "pred_class_id", "predicted_class",
        "y_pred", "basin_pred", "label_pred", "model_label", "predicted_basin", "ensemble_pred", "prediction", "predicted", "pred", "preds", "argmax",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not infer prediction column. Set PREDICTION_COLUMN near the top of the script to your saved class-argmax column.")


def map_prediction_labels(values):
    original = pd.Series(values)
    raw = pd.to_numeric(original, errors="coerce")
    invalid_text = original.notna() & raw.isna()
    if invalid_text.any():
        raise ValueError(f"Prediction labels must be numeric IDs; found nonnumeric values {original[invalid_text].astype(str).unique()[:5].tolist()}")
    mapped = raw.map(PRED_LABEL_MAP)
    finite_raw = raw.notna()
    unknown = finite_raw & mapped.isna()
    if unknown.any():
        found = sorted(raw[unknown].unique().tolist())
        raise ValueError(f"Prediction values {found} are not covered by PRED_LABEL_MAP={PRED_LABEL_MAP}. If predictions already use physical labels 1,2,3, set PRED_LABEL_MAP={{1: 1, 2: 2, 3: 3}}.")
    return mapped.to_numpy(dtype=float)


def transition_boundary(labels, valid_mask):
    labels = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if labels.shape != valid.shape or labels.ndim != 2:
        raise ValueError("labels and valid_mask must be same-shaped 2D arrays")
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    vertical = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    return boundary


def count_boxes_one_origin(mask, box_size, offset_y, offset_x):
    height, width = mask.shape
    count = 0
    for y0 in range(-offset_y, height, box_size):
        y1 = min(y0 + box_size, height)
        ys = max(y0, 0)
        if y1 <= ys:
            continue
        for x0 in range(-offset_x, width, box_size):
            x1 = min(x0 + box_size, width)
            xs = max(x0, 0)
            if x1 > xs and mask[ys:y1, xs:x1].any():
                count += 1
    return count


def count_boxes(mask, box_size):
    offsets = sorted(set([(0, 0), (box_size // 2, 0), (0, box_size // 2), (box_size // 2, box_size // 2)]))
    values = [count_boxes_one_origin(mask, box_size, oy, ox) for oy, ox in offsets]
    return float(np.mean(values))


def fit_box_counts(mask, res):
    scale_map = {}
    for fraction in BOX_FRACTIONS:
        box_size = max(2, int(round(float(fraction) * res)))
        scale_map[box_size] = box_size / float(res)
    sizes = sorted(scale_map)
    counts = np.asarray([count_boxes(mask, size) for size in sizes], dtype=float)
    fractions = np.asarray([scale_map[size] for size in sizes], dtype=float)
    usable = np.isfinite(counts) & (counts > 0) & np.isfinite(fractions) & (fractions > 0)
    sizes = np.asarray(sizes)[usable]
    counts = counts[usable]
    fractions = fractions[usable]
    if len(counts) < 4:
        return {"dimension": np.nan, "r2": np.nan, "drop_small": np.nan, "drop_large": np.nan}, []

    x = np.log(1.0 / fractions)
    y = np.log(counts)

    def regression(xv, yv):
        if len(xv) < 3:
            return np.nan, np.nan
        slope, intercept = np.polyfit(xv, yv, 1)
        predicted = slope * xv + intercept
        ss_res = float(np.sum((yv - predicted) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(slope), float(r2)

    full, r2 = regression(x, y)
    drop_small, _ = regression(x[1:], y[1:])
    drop_large, _ = regression(x[:-1], y[:-1])
    details = [
        {"box_pixels": int(size), "box_fraction": float(frac), "count_mean_over_origins": float(count), "log_inverse_scale": float(xx), "log_count": float(yy)}
        for size, frac, count, xx, yy in zip(sizes, fractions, counts, x, y)
    ]
    return {"dimension": full, "r2": r2, "drop_small": drop_small, "drop_large": drop_large}, details


def run_dbox_sweep():
    rows = []
    scale_rows = []
    for c_value in C_VALUES:
        for res in RES_LIST:
            path = DATA_DIR / f"{c_value:.2f}" / RAW_FILE_TEMPLATE.format(res=res)
            if not path.exists():
                raise FileNotFoundError(f"Missing raw reference grid: {path}")
            frame = pd.read_csv(path)
            if not {"x0", "vx0", "label_id"}.issubset(frame.columns):
                raise ValueError(f"{path} must contain x0, vx0, label_id")
            x_grid, vx_grid, ix, iy = validated_grid_indices(frame, res, allow_missing=True)

            true_grid = np.full((res, res), -999, dtype=int)
            true_grid[iy, ix] = pd.to_numeric(frame["label_id"], errors="raise").astype(int).to_numpy()
            valid_true = np.isin(true_grid, list(VALID_LABELS))

            models, mean, scale = load_ensemble(c_value, res)
            xx, vv = np.meshgrid(x_grid, vx_grid)
            grid_points = np.column_stack([xx.ravel(), vv.ravel()])
            per_model = predict_ensemble(models, grid_points, mean, scale)
            pred_grid = per_model.mean(axis=1).argmax(axis=1).reshape(res, res)
            del models

            true_boundary = transition_boundary(true_grid, valid_true)
            learned_boundary = transition_boundary(pred_grid, valid_true)
            true_fit, true_details = fit_box_counts(true_boundary, res)
            learned_fit, learned_details = fit_box_counts(learned_boundary, res)
            rows.append({
                "C": c_value,
                "resolution": res,
                "input_file": str(path),
                "n_reference_rows": int(len(frame)),
                "n_valid_reference_cells": int(valid_true.sum()),
                "n_missing_grid_cells": int(res * res - len(frame)),
                "reference_boundary_cells": int(true_boundary.sum()),
                "learned_boundary_cells_on_reference_valid_domain": int(learned_boundary.sum()),
                "D_true_box": true_fit["dimension"],
                "D_true_R2": true_fit["r2"],
                "D_true_drop_smallest_box": true_fit["drop_small"],
                "D_true_drop_largest_box": true_fit["drop_large"],
                "D_learned_box": learned_fit["dimension"],
                "D_learned_R2": learned_fit["r2"],
                "D_learned_drop_smallest_box": learned_fit["drop_small"],
                "D_learned_drop_largest_box": learned_fit["drop_large"],
            })
            for method, details in (("true", true_details), ("learned", learned_details)):
                for detail in details:
                    scale_rows.append({"C": c_value, "resolution": res, "method": method, **detail})
            print(f"D_box C={c_value:.2f} res={res}: true={true_fit['dimension']:.4f}, learned={learned_fit['dimension']:.4f}")
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "dbox_vs_resolution.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "dbox_scale_counts.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, c_value in zip(axes.ravel(), C_VALUES):
        sub = results[results["C"] == c_value].sort_values("resolution")
        ax.plot(sub["resolution"], sub["D_true_box"], marker="o", label="Reference boundary")
        ax.plot(sub["resolution"], sub["D_learned_box"], marker="s", linestyle="--", label="Learned boundary")
        ax.set_title(f"C = {c_value:.2f}")
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel("Finite-scale box-counting slope")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Reference vs learned boundary dimension across resolution")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dbox_vs_resolution.png", dpi=300, bbox_inches="tight")
    with open(OUT_DIR / "dbox_vs_resolution.pkl", "wb") as handle:
        pickle.dump(fig, handle)
    plt.close(fig)
    return results


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_ensemble(c_value, res):
    save_dir = MODEL_DIR / f"{c_value:.2f}" / f"{res}x{res}"
    pre_path = save_dir / "preprocessing.pt"
    if not pre_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {pre_path}")
    pre = load_torch_checkpoint(pre_path)
    mean = np.asarray(pre["scaler_mean"], dtype=np.float64).reshape(1, -1)
    scale = np.asarray(pre["scaler_scale"], dtype=np.float64).reshape(1, -1)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Invalid scaler in {pre_path}")
    models = []
    for i in range(5):
        model_path = save_dir / f"MODEL_{i}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
        model = BasinClassifier(hidden=256, ff_scale=5.0, ff_seed=42 + int(round(c_value * 100)) + res + i + 1).to(DEVICE)
        model.load_state_dict(load_torch_checkpoint(model_path))
        model.eval()
        models.append(model)
    return models, mean, scale


def predict_ensemble(models, x_raw, mean, scale, batch_size=1024):
    x_raw = np.asarray(x_raw, dtype=np.float32)
    x_std = ((x_raw - mean) / scale).astype(np.float32)
    result = []
    with torch.no_grad():
        for start in range(0, len(x_std), batch_size):
            x_t = torch.as_tensor(x_std[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            per_model = [F.softmax(model(x_t), dim=1).detach().cpu().numpy() for model in models]
            result.append(np.stack(per_model, axis=1))
    if not result:
        return np.empty((0, len(models), 3), dtype=np.float64)
    return np.concatenate(result, axis=0)


def entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=-1)


def fit_slope(epsilons, response, min_response=MIN_RESPONSE):
    epsilons = np.asarray(epsilons, dtype=float)
    response = np.asarray(response, dtype=float)
    valid = np.isfinite(epsilons) & np.isfinite(response) & (epsilons > 0) & (response > min_response)
    if valid.sum() < 3:
        return np.nan, np.nan, valid
    x = np.log(epsilons[valid])
    y = np.log(response[valid])
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2), valid


def slope_on_fixed_mask(epsilons, response, mask):
    eps = np.asarray(epsilons, dtype=float)[mask]
    values = np.asarray(response, dtype=float)[mask]
    if len(values) < 3 or np.any(~np.isfinite(values)) or np.any(values <= MIN_RESPONSE):
        return np.nan
    return float(np.polyfit(np.log(eps), np.log(values), 1)[0])


def sample_uniform_grid_blocks(frame, ix, iy, res, rng, n_target):
    block_cols = int(np.ceil(res / BLOCK_PIXELS))
    block_ids_all = (iy // BLOCK_PIXELS) * block_cols + (ix // BLOCK_PIXELS)
    unique_blocks = np.unique(block_ids_all)
    target_blocks = min(200, len(unique_blocks), n_target)
    quotient, remainder = divmod(n_target, target_blocks)
    selected_rows = []
    selected_blocks = []
    block_order = rng.choice(unique_blocks, size=target_blocks, replace=False)
    for index, block in enumerate(block_order):
        members = np.flatnonzero(block_ids_all == block)
        n_take = quotient + (1 if index < remainder else 0)
        n_take = min(n_take, len(members))
        chosen = rng.choice(members, size=n_take, replace=False)
        selected_rows.extend(chosen.tolist())
        selected_blocks.extend([int(block)] * len(chosen))
    selected_rows = np.asarray(selected_rows, dtype=int)
    selected_blocks = np.asarray(selected_blocks, dtype=int)
    if len(selected_rows) < 30:
        raise ValueError("Spatial block sampling produced too few initial conditions")
    return selected_rows, selected_blocks


def block_bootstrap_positions(block_ids, rng):
    unique_blocks = np.unique(block_ids)
    members = {block: np.flatnonzero(block_ids == block) for block in unique_blocks}
    draws = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
    return np.concatenate([members[block] for block in draws])


def compute_pointwise_responses(models, x_raw, mean, scale, epsilons, rng):
    n = len(x_raw)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    base_probs = predict_ensemble(models, x_raw, mean, scale)
    base_mean = base_probs.mean(axis=1)
    base_pred = base_mean.argmax(axis=1)
    base_epi = np.maximum(entropy(base_mean) - entropy(base_probs).mean(axis=1), 0.0)
    flip = np.zeros((len(epsilons), n), dtype=float)
    dp = np.zeros_like(flip)
    du = np.zeros_like(flip)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    for e_idx, epsilon in enumerate(epsilons):
        perturbed = x_raw + float(epsilon) * directions
        perturbed_probs = predict_ensemble(models, perturbed, mean, scale)
        perturbed_mean = perturbed_probs.mean(axis=1)
        perturbed_pred = perturbed_mean.argmax(axis=1)
        perturbed_epi = np.maximum(entropy(perturbed_mean) - entropy(perturbed_probs).mean(axis=1), 0.0)
        flip[e_idx] = (perturbed_pred != base_pred).astype(float)
        dp[e_idx] = np.abs(perturbed_mean - base_mean).sum(axis=1)
        du[e_idx] = np.abs(perturbed_epi - base_epi)
    return {"flip": flip, "dP": dp, "dU": du}


def run_beta_bootstrap():
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    curve_rows = []
    for c_value in C_VALUES:
        for res in RES_LIST:
            path = DATA_DIR / f"{c_value:.2f}" / RAW_FILE_TEMPLATE.format(res=res)
            if not path.exists():
                raise FileNotFoundError(f"Missing raw grid for beta bootstrap: {path}")
            frame = pd.read_csv(path)
            if not {"x0", "vx0", "label_id"}.issubset(frame.columns):
                raise ValueError(f"{path} must contain x0, vx0, label_id")
            _, _, ix, iy = validated_grid_indices(frame, res, allow_missing=True)
            frame = frame[np.isin(pd.to_numeric(frame["label_id"], errors="raise").to_numpy(), list(VALID_LABELS))].reset_index(drop=True)
            _, _, ix, iy = validated_grid_indices(frame, res, allow_missing=True)
            n_sample = min(N_PERTURB, len(frame))
            selected_rows, selected_blocks = sample_uniform_grid_blocks(frame, ix, iy, res, rng, n_sample)
            x_raw = frame.loc[selected_rows, ["x0", "vx0"]].to_numpy(dtype=np.float64)
            models, mean, scale = load_ensemble(c_value, res)
            pointwise = compute_pointwise_responses(models, x_raw, mean, scale, EPSILONS, rng)
            for metric in ("flip", "dP", "dU"):
                curve = pointwise[metric].mean(axis=1)
                slope, r2, fit_mask = fit_slope(EPSILONS, curve)
                bootstrap_slopes = np.full(N_BOOT, np.nan, dtype=float)
                for b in range(N_BOOT):
                    positions = block_bootstrap_positions(selected_blocks, rng)
                    boot_curve = pointwise[metric][:, positions].mean(axis=1)
                    bootstrap_slopes[b] = slope_on_fixed_mask(EPSILONS, boot_curve, fit_mask)
                finite_boot = bootstrap_slopes[np.isfinite(bootstrap_slopes)]
                if len(finite_boot) >= max(30, int(0.5 * N_BOOT)):
                    ci_low, ci_high = np.percentile(finite_boot, [2.5, 97.5])
                else:
                    ci_low, ci_high = np.nan, np.nan
                summary_rows.append({
                    "C": c_value,
                    "resolution": res,
                    "metric": metric,
                    "beta": slope,
                    "fit_R2": r2,
                    "ci95_low_spatial_block_bootstrap": ci_low,
                    "ci95_high_spatial_block_bootstrap": ci_high,
                    "n_boot_replicates": int(N_BOOT),
                    "n_valid_boot_slopes": int(len(finite_boot)),
                    "n_initial_conditions": int(len(selected_rows)),
                    "n_spatial_blocks": int(len(np.unique(selected_blocks))),
                    "sampling_design": "uniform grid; equal-size spatial-block subsample; fixed trained ensemble",
                    "block_pixels": int(BLOCK_PIXELS),
                    "fit_epsilon_min": float(EPSILONS[fit_mask].min()) if fit_mask.any() else np.nan,
                    "fit_epsilon_max": float(EPSILONS[fit_mask].max()) if fit_mask.any() else np.nan,
                })
                for e_idx, epsilon in enumerate(EPSILONS):
                    curve_rows.append({"C": c_value, "resolution": res, "metric": metric, "epsilon": float(epsilon), "response_mean": float(curve[e_idx])})
            print(f"Beta bootstrap C={c_value:.2f} res={res}: done using {n_sample} initial conditions")
            del models
    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    summary.to_csv(OUT_DIR / "beta_bootstrap_summary.csv", index=False)
    curves.to_csv(OUT_DIR / "beta_response_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, metric in zip(axes, ["flip", "dP", "dU"]):
        sub = summary[summary["metric"] == metric]
        for c_value in C_VALUES:
            group = sub[sub["C"] == c_value].sort_values("resolution")
            y = group["beta"].to_numpy(dtype=float)
            low = group["ci95_low_spatial_block_bootstrap"].to_numpy(dtype=float)
            high = group["ci95_high_spatial_block_bootstrap"].to_numpy(dtype=float)
            lower = np.where(np.isfinite(low), np.maximum(0.0, y - low), 0.0)
            upper = np.where(np.isfinite(high), np.maximum(0.0, high - y), 0.0)
            ax.errorbar(group["resolution"], y, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=f"C={c_value:.2f}")
        ax.axhline(1.0, linestyle="--", linewidth=1, label="β = 1" if metric == "dU" else None)
        ax.set_title(f"β for {metric}")
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel("Fitted exponent")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Perturbation-response exponents with spatial-block bootstrap intervals")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_bootstrap_by_resolution.png", dpi=300, bbox_inches="tight")
    with open(OUT_DIR / "beta_bootstrap_by_resolution.pkl", "wb") as handle:
        pickle.dump(fig, handle)
    plt.close(fig)
    return summary


def infer_probability_columns(frame):
    if PROBABILITY_COLUMNS is not None:
        if len(PROBABILITY_COLUMNS) != 3 or any(col not in frame.columns for col in PROBABILITY_COLUMNS):
            raise ValueError(f"Set PROBABILITY_COLUMNS to three existing columns; got {PROBABILITY_COLUMNS}")
        return list(PROBABILITY_COLUMNS)
    prefixes = ["p_mean_", "mean_prob_", "prob_mean_", "pred_prob_", "class_prob_", "prob_class_", "probability_", "proba_", "prob_", "p_", "probability", "proba", "prob", "p"]
    groups = {}
    for col in frame.columns:
        lowered = str(col).strip().lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                suffix = lowered[len(prefix):]
                if suffix.isdigit():
                    groups.setdefault(prefix, {})[int(suffix)] = col
                    break
    candidates = []
    for prefix, group in groups.items():
        if len(group) == 3:
            indices = sorted(group)
            if indices in ([0, 1, 2], [1, 2, 3]):
                candidates.append((0 if indices == [0, 1, 2] else 1, prefix, [group[i] for i in indices]))
    if not candidates:
        raise ValueError("Could not infer three stored ensemble-mean probability columns. Set PROBABILITY_COLUMNS = ['col0','col1','col2'] at the top of the script.")
    candidates.sort(key=lambda item: (item[0], -len(item[1])))
    best_rank = candidates[0][0]
    best = [item for item in candidates if item[0] == best_rank]
    if len(best) > 1:
        unique_columns = {tuple(item[2]) for item in best}
        if len(unique_columns) > 1:
            raise ValueError(f"Multiple plausible probability-column groups found: {[item[2] for item in best]}. Set PROBABILITY_COLUMNS explicitly.")
    return candidates[0][2]


def safe_binary_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return {"AUROC": np.nan, "average_precision": np.nan, "precision_at_top10pct": np.nan, "recall_at_top10pct": np.nan}
    auc = roc_auc_score(y_true.astype(int), scores)
    ap = average_precision_score(y_true.astype(int), scores)
    k = max(1, int(np.ceil(0.10 * len(scores))))
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[np.argsort(scores, kind="mergesort")[-k:]] = True
    precision = precision_score(y_true, chosen, zero_division=0)
    recall = recall_score(y_true, chosen, zero_division=0)
    return {"AUROC": float(auc), "average_precision": float(ap), "precision_at_top10pct": float(precision), "recall_at_top10pct": float(recall)}


def run_mi_localization():
    raw_path = DATA_DIR / f"{MI_C_VALUE:.2f}" / RAW_FILE_TEMPLATE.format(res=MI_RESOLUTION)
    if not raw_path.exists():
        raise FileNotFoundError(f"MI localization needs the raw full-grid reference file: {raw_path}")
    frame = pd.read_csv(raw_path)
    required = {"x0", "vx0", "label_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{raw_path} must contain columns {sorted(required)}")
    res = MI_RESOLUTION
    x_grid, vx_grid, ix, iy = validated_grid_indices(frame, res, allow_missing=True)

    true_grid = np.full((res, res), -999, dtype=int)
    true_grid[iy, ix] = pd.to_numeric(frame["label_id"], errors="raise").astype(int).to_numpy()
    valid_grid = np.isin(true_grid, list(VALID_LABELS))
    boundary_grid = transition_boundary(true_grid, valid_grid)

    models, mean, scale = load_ensemble(MI_C_VALUE, res)
    xx, vv = np.meshgrid(x_grid, vx_grid)
    grid_points = np.column_stack([xx.ravel(), vv.ravel()])
    per_model = predict_ensemble(models, grid_points, mean, scale)
    p_mean = per_model.mean(axis=1)
    mi_values = entropy(p_mean) - entropy(per_model).mean(axis=1)
    mi_values = np.maximum(mi_values, 0.0)
    entropy_values = entropy(p_mean)
    prob_grid = p_mean.reshape(res, res, 3)
    mi_grid = mi_values.reshape(res, res)
    entropy_grid = entropy_values.reshape(res, res)

    dx = float(x_grid[1] - x_grid[0])
    dvx = float(vx_grid[1] - vx_grid[0])
    grad_x = np.gradient(prob_grid, dx, axis=1)
    grad_vx = np.gradient(prob_grid, dvx, axis=0)
    gradient_grid = np.sqrt(np.sum(grad_x ** 2 + grad_vx ** 2, axis=2))

    eval_mask = valid_grid & np.isfinite(mi_grid) & np.isfinite(entropy_grid) & np.isfinite(gradient_grid)
    if eval_mask.sum() == 0:
        raise ValueError("No valid cells remain after applying the shared finite-score mask")
    y_true = boundary_grid[eval_mask]
    score_sets = {
        "Epistemic MI": mi_grid[eval_mask],
        "Predictive entropy": entropy_grid[eval_mask],
        "Probability-gradient norm": gradient_grid[eval_mask],
    }
    rows = []
    for name, scores in score_sets.items():
        metrics = safe_binary_metrics(y_true, scores)
        rows.append({
            "C": MI_C_VALUE,
            "resolution": res,
            "method": name,
            "source_file": str(raw_path),
            "uncertainty_source": "recomputed from saved five-model ensemble over full expected grid",
            "n_reference_rows": int(len(frame)),
            "n_evaluated_cells": int(len(y_true)),
            "boundary_prevalence": float(y_true.mean()),
            "random_ranking_AP_baseline": float(y_true.mean()),
            **metrics,
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / f"mi_localization_metrics_C{MI_C_VALUE:.2f}_{res}x{res}.csv", index=False)
    metric_names = ["AUROC", "average_precision", "precision_at_top10pct", "recall_at_top10pct"]
    labels = ["AUROC", "Average precision", "Precision @ top 10%", "Recall @ top 10%"]
    methods = list(score_sets)
    x = np.arange(len(metric_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        row = results[results["method"] == method].iloc[0]
        values = [row[name] for name in metric_names]
        ax.bar(x + (i - 1) * width, values, width, label=method)
    prevalence = float(y_true.mean())
    ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1, label=f"Boundary prevalence / random AP={prevalence:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Boundary localization at C={MI_C_VALUE:.2f}, {res}×{res}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    stem = OUT_DIR / f"mi_localization_C{MI_C_VALUE:.2f}_{res}x{res}"
    fig.savefig(Path(str(stem) + ".png"), dpi=300, bbox_inches="tight")
    with open(Path(str(stem) + ".pkl"), "wb") as handle:
        pickle.dump(fig, handle)
    plt.close(fig)
    print(results.to_string(index=False))
    return results


def main():
    print(f"Device: {DEVICE}")
    errors = []
    if RUN_DBOX_SWEEP:
        try:
            run_dbox_sweep()
        except Exception as exc:
            errors.append(("D_box sweep", exc))
            print(f"D_box sweep failed: {exc}")
    if RUN_BETA_BOOTSTRAP:
        try:
            run_beta_bootstrap()
        except Exception as exc:
            errors.append(("beta bootstrap", exc))
            print(f"Beta bootstrap failed: {exc}")
    if RUN_MI_LOCALIZATION:
        try:
            run_mi_localization()
        except Exception as exc:
            errors.append(("MI localization", exc))
            print(f"MI localization failed: {exc}")
    if errors:
        summary = "; ".join(f"{name}: {error}" for name, error in errors)
        raise RuntimeError(f"One or more analyses failed. Check configuration and input schemas. {summary}")
    print(f"All enabled analyses completed. Outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
