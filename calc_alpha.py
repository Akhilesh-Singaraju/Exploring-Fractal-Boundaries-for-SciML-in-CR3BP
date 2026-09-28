import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def _compute_uncertainty_numba(x, vx, labels, idx, epsilons, x_min, vx_min, dx, dvx, num_x, num_vx):
    num_samples = len(idx)
    num_eps = len(epsilons)
    f_eps = np.zeros(num_eps, dtype=np.float64)
    num_labels = len(labels)

    for e_idx in prange(num_eps):
        eps = epsilons[e_idx]
        uncertain_count = 0

        for k in range(num_samples):
            i = idx[k]
            x0 = x[i]
            vx0 = vx[i]
            l0 = labels[i]

            theta = np.random.uniform(0.0, 2.0 * np.pi)
            x_p = x0 + eps * np.cos(theta)
            vx_p = vx0 + eps * np.sin(theta)

            ix = int(round((x_p - x_min) / dx))
            if ix < 0:
                ix = 0
            elif ix >= num_x:
                ix = num_x - 1

            ivx = int(round((vx_p - vx_min) / dvx))
            if ivx < 0:
                ivx = 0
            elif ivx >= num_vx:
                ivx = num_vx - 1

            p_idx = ivx * num_x + ix
            if p_idx >= num_labels:
                p_idx = num_labels - 1

            if labels[p_idx] != l0:
                uncertain_count += 1

        f_eps[e_idx] = uncertain_count / num_samples

    return f_eps


def compute_physical_alpha(csv_path, num_samples=3000, epsilons=None):
    if epsilons is None:
        epsilons = np.logspace(-3.5, -1.0, 8)

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return np.nan, np.nan

    df = pd.read_csv(csv_path)

    def get_col(df, options, fallback_idx):
        for name in options:
            if name in df.columns:
                return df[name].values
        return df.iloc[:, fallback_idx].values

    x_raw = get_col(df, ['x', 'x0'], 0)
    vx_raw = get_col(df, ['vx', 'vx0'], 1)
    label_raw = get_col(df, ['label', 'label_id'], 2)

    x = np.ascontiguousarray(x_raw, dtype=np.float64)
    vx = np.ascontiguousarray(vx_raw, dtype=np.float64)

    if label_raw.dtype == object:
        labels = pd.factorize(label_raw)[0].astype(np.int64)
    else:
        labels = np.ascontiguousarray(label_raw, dtype=np.int64)

    unique_x = np.unique(x)
    unique_vx = np.unique(vx)

    dx = np.min(np.diff(unique_x))
    dvx = np.min(np.diff(unique_vx))

    x_min = x.min()
    vx_min = vx.min()
    num_x = len(unique_x)
    num_vx = len(unique_vx)

    sample_size = min(num_samples, len(df))
    idx = np.random.choice(len(df), size=sample_size, replace=False)

    f_eps = _compute_uncertainty_numba(
        x, vx, labels, idx, epsilons, x_min, vx_min, dx, dvx, num_x, num_vx
    )

    valid = f_eps > 0
    if not np.any(valid):
        return np.nan, np.nan

    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(epsilons[valid]), np.log(f_eps[valid])
    )

    return slope, r_value**2


if __name__ == "__main__":
    C_values = [3.00, 3.05, 3.10, 3.14, 3.20]
    results = {}

    for C in C_values:
        path = f"files/basin_C{C:.2f}.csv"
        alpha, r2 = compute_physical_alpha(path)
        if not np.isnan(alpha):
            results[C] = {"alpha": alpha, "r2": r2}
            print(f"C = {C:.2f} | Physical Alpha = {alpha:.4f} | R^2 = {r2:.4f}")