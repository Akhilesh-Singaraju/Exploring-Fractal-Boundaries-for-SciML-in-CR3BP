import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from numba import njit, prange

PHYSICAL_LABELS = (1, 2, 3)
OUT_CSV = os.path.join("files", "figures", "alpha_results.csv")


@njit(fastmath=True, parallel=True)
def _uncertainty_fraction(
    x,
    vx,
    labels,
    idx,
    epsilons,
    thetas,
    labels_grid,
    x_min,
    vx_min,
    dx,
    dvx,
    num_x,
    num_vx,
):
    num_samples = idx.shape[0]
    num_eps = epsilons.shape[0]
    f_eps = np.zeros(num_eps, dtype=np.float64)

    for e_idx in prange(num_eps):
        eps = epsilons[e_idx]
        uncertain = 0
        valid = 0

        for k in range(num_samples):
            i = idx[k]
            l0 = labels[i]

            th = thetas[e_idx, k]
            xp = x[i] + eps * np.cos(th)
            vxp = vx[i] + eps * np.sin(th)

            ix = int(round((xp - x_min) / dx))
            ivx = int(round((vxp - vx_min) / dvx))

            if ix < 0 or ix >= num_x or ivx < 0 or ivx >= num_vx:
                continue

            lp = labels_grid[ivx, ix]

            if lp < 1 or lp > 3:
                continue

            valid += 1
            if lp != l0:
                uncertain += 1

        if valid > 0:
            f_eps[e_idx] = uncertain / valid

    return f_eps


def _get_numeric(df, names, fallback_col):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
    return pd.to_numeric(df.iloc[:, fallback_col], errors="coerce").to_numpy(dtype=np.float64)


def _build_grid(x, vx, labels):
    ux = np.unique(x)
    uv = np.unique(vx)

    num_x = ux.size
    num_vx = uv.size

    x_min = float(ux[0])
    x_max = float(ux[-1])
    vx_min = float(uv[0])
    vx_max = float(uv[-1])

    dx = (x_max - x_min) / (num_x - 1) if num_x > 1 else 1.0
    dvx = (vx_max - vx_min) / (num_vx - 1) if num_vx > 1 else 1.0

    if not np.isfinite(dx) or dx <= 0.0:
        dx = 1.0
    if not np.isfinite(dvx) or dvx <= 0.0:
        dvx = 1.0

    labels_grid = np.full((num_vx, num_x), -1, dtype=np.int64)

    ix = np.rint((x - x_min) / dx).astype(np.int64)
    ivx = np.rint((vx - vx_min) / dvx).astype(np.int64)

    ix = np.clip(ix, 0, num_x - 1)
    ivx = np.clip(ivx, 0, num_vx - 1)

    labels_grid[ivx, ix] = labels

    return labels_grid, x_min, vx_min, dx, dvx, num_x, num_vx, x_max, vx_max


def _default_epsilons(x_min, x_max, vx_min, vx_max, dx, dvx):
    grid_scale = max(float(dx), float(dvx), 1.0e-12)

    x_span = max(float(x_max - x_min), grid_scale)
    vx_span = max(float(vx_max - vx_min), grid_scale)

    domain_scale = max(min(x_span, vx_span), grid_scale)
    hi_cap = 0.5 * domain_scale

    lo = 5.0 * grid_scale
    if lo >= hi_cap:
        lo = hi_cap / 50.0

    hi = min(0.2 * domain_scale, hi_cap)

    if hi <= lo:
        hi = hi_cap

    if hi / lo < 10.0:
        hi = min(10.0 * lo, hi_cap)

    if hi <= lo:
        lo = hi_cap / 100.0
        hi = hi_cap

    if hi <= lo:
        lo = 0.005
        hi = 0.05

    return np.logspace(np.log10(lo), np.log10(hi), 20)


def _fit_alpha(epsilons, f_eps):
    eps = np.asarray(epsilons, dtype=np.float64)
    f = np.asarray(f_eps, dtype=np.float64)

    base_mask = np.isfinite(eps) & np.isfinite(f) & (eps > 0.0) & (f > 0.0)

    mask = base_mask & (f < 0.95)
    if int(np.sum(mask)) < 3:
        mask = base_mask

    n = int(np.sum(mask))

    if n < 3:
        return {
            "alpha": np.nan,
            "stderr": np.nan,
            "r2": np.nan,
            "chi2": np.nan,
            "n": 0,
            "eps_min": np.nan,
            "eps_max": np.nan,
        }

    log_eps = np.log(eps[mask])
    log_f = np.log(f[mask])

    if np.ptp(log_eps) <= 0.0:
        return {
            "alpha": np.nan,
            "stderr": np.nan,
            "r2": np.nan,
            "chi2": np.nan,
            "n": n,
            "eps_min": float(eps[mask].min()),
            "eps_max": float(eps[mask].max()),
        }

    res = linregress(log_eps, log_f)

    residuals = log_f - (res.slope * log_eps + res.intercept)
    dof = max(n - 2, 1)
    chi2 = float(np.sum(residuals**2) / dof)

    return {
        "alpha": float(res.slope),
        "stderr": float(res.stderr) if res.stderr is not None else np.nan,
        "r2": float(res.rvalue**2),
        "chi2": chi2,
        "n": n,
        "eps_min": float(eps[mask].min()),
        "eps_max": float(eps[mask].max()),
    }


def _nan_row(C):
    return {
        "C": C,
        "alpha": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "r_squared": np.nan,
        "reduced_chi2": np.nan,
        "stderr_alpha": np.nan,
        "n_fit_points": 0,
        "n_bootstrap": 0,
        "eps_min": np.nan,
        "eps_max": np.nan,
    }


def compute_uncertainty_alpha(
    csv_path,
    C,
    num_samples=10000,
    n_bootstrap=200,
    seed=42,
    max_eps_fit=0.2,
):
    row = _nan_row(C)

    if not os.path.exists(csv_path):
        return row

    df = pd.read_csv(csv_path)

    x = _get_numeric(df, ["x0", "x"], 0)
    vx = _get_numeric(df, ["vx0", "vx"], 1)
    x = (x - x.min()) / (x.max() - x.min())
    vx = (vx - vx.min()) / (vx.max() - vx.min())
    lab = _get_numeric(df, ["label_id", "label"], 3)

    finite = np.isfinite(x) & np.isfinite(vx) & np.isfinite(lab)

    x = x[finite]
    vx = vx[finite]
    labels = lab[finite].astype(np.int64)

    if x.size < 10:
        return row

    phys_mask = np.isin(labels, np.array(PHYSICAL_LABELS, dtype=np.int64))
    phys_idx = np.nonzero(phys_mask)[0]
    n_phys = phys_idx.size

    if n_phys == 0:
        return row

    try:
        (
            labels_grid,
            x_min,
            vx_min,
            dx,
            dvx,
            num_x,
            num_vx,
            x_max,
            vx_max,
        ) = _build_grid(x, vx, labels)
    except Exception:
        return row

    epsilons = _default_epsilons(x_min, x_max, vx_min, vx_max, dx, dvx)

    sample_size = min(int(num_samples), n_phys)
    if sample_size <= 0:
        return row

    rng = np.random.default_rng(seed)

    idx = phys_idx[rng.choice(n_phys, size=sample_size, replace=False)]
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(epsilons.size, sample_size))

    f_eps = _uncertainty_fraction(
        x,
        vx,
        labels,
        idx,
        epsilons,
        thetas,
        labels_grid,
        x_min,
        vx_min,
        dx,
        dvx,
        num_x,
        num_vx,
    )

    print(f_eps)

    fit_mask = epsilons <= max_eps_fit
    if np.sum(fit_mask) < 3:
        fit_mask = np.ones_like(epsilons, dtype=bool)
        
    eps_fit = epsilons[fit_mask]
    f_fit = f_eps[fit_mask]

    fit = _fit_alpha(eps_fit, f_fit) 

    row.update(
        {
            "alpha": fit["alpha"],
            "r_squared": fit["r2"],
            "reduced_chi2": fit["chi2"],
            "stderr_alpha": fit["stderr"],
            "n_fit_points": fit["n"],
            "eps_min": fit["eps_min"],
            "eps_max": fit["eps_max"],
        }
    )

    if not np.isfinite(fit["alpha"]) or n_bootstrap <= 0:
        return row

    bootstrap_alphas = []

    for b in range(int(n_bootstrap)):
        brng = np.random.default_rng(seed + 10_000 + b)

        idx_b = phys_idx[brng.choice(n_phys, size=sample_size, replace=True)]
        thetas_b = brng.uniform(0.0, 2.0 * np.pi, size=(epsilons.size, sample_size))

        f_b = _uncertainty_fraction(
            x,
            vx,
            labels,
            idx_b,
            epsilons,
            thetas_b,
            labels_grid,
            x_min,
            vx_min,
            dx,
            dvx,
            num_x,
            num_vx,
        )
        f_b_fit = f_b[fit_mask]
        fit_b = _fit_alpha(eps_fit, f_b_fit)

        if np.isfinite(fit_b["alpha"]):
            bootstrap_alphas.append(fit_b["alpha"])

    bootstrap_alphas = np.asarray(bootstrap_alphas, dtype=np.float64)

    row["n_bootstrap"] = len(bootstrap_alphas)

    if len(bootstrap_alphas) >= 10:
        row["ci_lower"] = float(np.percentile(bootstrap_alphas, 2.5))
        row["ci_upper"] = float(np.percentile(bootstrap_alphas, 97.5))

    return row


if __name__ == "__main__":
    C_values = [3.00, 3.05, 3.10, 3.14, 3.20]

    NUM_SAMPLES = 10000
    N_BOOTSTRAP = 200

    rows = []

    # df = pd.read_csv("files/data/3.00/700x700.csv")
    
    # x = pd.to_numeric(df["x0"], errors="coerce")
    # vx = pd.to_numeric(df["vx0"], errors="coerce")
    # labels = pd.to_numeric(df["label_id"], errors="coerce")

    # valid = x.notna() & vx.notna() & labels.notna()
    # df = df.loc[valid].copy()

    # print("Rows:", len(df))
    # print("Unique x:", df["x0"].nunique())
    # print("Unique vx:", df["vx0"].nunique())
    # print("Unique (x, vx) pairs:", df[["x0", "vx0"]].drop_duplicates().shape[0])
    # print("Expected grid cells:", df["x0"].nunique() * df["vx0"].nunique())
    # print("Physical-label fraction:", df["label_id"].isin([1, 2, 3]).mean())

    for C in C_values:
        path = f"files/data/{C:.2f}/700x700.csv"

        rows.append(
            compute_uncertainty_alpha(
                path,
                C=C,
                num_samples=NUM_SAMPLES,
                n_bootstrap=N_BOOTSTRAP,
                seed=42,
            )
        )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print_df = df.copy()
    print_df["ci_95"] = (
        "["
        + print_df["ci_lower"].map(lambda z: f"{z:.4f}")
        + ", "
        + print_df["ci_upper"].map(lambda z: f"{z:.4f}")
        + "]"
    )

    cols = [
        "C",
        "alpha",
        "ci_95",
        "r_squared",
        "reduced_chi2",
        "stderr_alpha",
        "n_fit_points",
        "n_bootstrap",
        "eps_min",
        "eps_max",
    ]

    print(print_df[cols].to_string(index=False, float_format=lambda z: f"{z: .6f}"))
    print(f"\nSaved: {OUT_CSV}")