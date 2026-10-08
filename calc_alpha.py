import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import linregress
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PHYSICAL_LABELS = (1, 2, 3)
PICKLE_DIR = "files/pickles"
FIGURE_DIR = "files/figures"


@njit(fastmath=True, parallel=True)
def _compute_uncertainty_numba(x, vx, labels, idx, epsilons, thetas,
                               labels_grid, x_min, vx_min, dx, dvx,
                               num_x, num_vx):
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
            x_p = x[i] + eps * np.cos(th)
            vx_p = vx[i] + eps * np.sin(th)

            ix = int(round((x_p - x_min) / dx))
            ivx = int(round((vx_p - vx_min) / dvx))

            if ix < 0:
                ix = 0
            elif ix >= num_x:
                ix = num_x - 1
            if ivx < 0:
                ivx = 0
            elif ivx >= num_vx:
                ivx = num_vx - 1

            l_p = labels_grid[ivx, ix]

            if 1 <= l_p <= 3:
                valid += 1
                if l_p != l0:
                    uncertain += 1

        f_eps[e_idx] = (uncertain / valid) if valid > 0 else 0.0

    return f_eps


def _build_grid(x, vx, labels):
    unique_x = np.unique(x)
    unique_vx = np.unique(vx)
    num_x = unique_x.shape[0]
    num_vx = unique_vx.shape[0]

    dx = (unique_x[-1] - unique_x[0]) / (num_x - 1)
    dvx = (unique_vx[-1] - unique_vx[0]) / (num_vx - 1)

    labels_grid = np.full((num_vx, num_x), -1, dtype=np.int64)
    ix_all = np.round((x - unique_x[0]) / dx).astype(np.int64)
    ivx_all = np.round((vx - unique_vx[0]) / dvx).astype(np.int64)
    labels_grid[ivx_all, ix_all] = labels

    return labels_grid, unique_x[0], unique_vx[0], dx, dvx, num_x, num_vx


def _fit_alpha(epsilons, f_eps):
    valid = f_eps > 0.0
    n_valid = valid.sum()

    if n_valid < 3:
        return {
            "alpha": np.nan, "intercept": np.nan,
            "r_squared": np.nan, "reduced_chi2": np.nan,
            "n_points": n_valid, "stderr_slope": np.nan,
        }

    log_eps = np.log(epsilons[valid])
    log_f = np.log(f_eps[valid])

    res = linregress(log_eps, log_f)

    residuals = log_f - (res.slope * log_eps + res.intercept)
    dof = max(n_valid - 2, 1)
    reduced_chi2 = np.sum(residuals ** 2) / dof

    return {
        "alpha": res.slope,
        "intercept": res.intercept,
        "r_squared": res.rvalue ** 2,
        "reduced_chi2": reduced_chi2,
        "n_points": n_valid,
        "stderr_slope": res.stderr,
    }


def compute_physical_alpha(csv_path, num_samples=3000, epsilons=None,
                           n_bootstrap=1000, seed=42, return_f=False):
    empty_result = {
        "alpha": np.nan, "ci_95": (np.nan, np.nan),
        "r_squared": np.nan, "reduced_chi2": np.nan,
        "stderr": np.nan, "n_bootstrap": 0,
        "bootstrap_alphas": np.array([]),
        "fit_info": {},
    }

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return empty_result, None

    df = pd.read_csv(csv_path)

    def get_col(options, fallback):
        for name in options:
            if name in df.columns:
                return df[name].values
        return df.iloc[:, fallback].values

    x = np.ascontiguousarray(get_col(['x0', 'x'], 0), dtype=np.float64)
    vx = np.ascontiguousarray(get_col(['vx0', 'vx'], 1), dtype=np.float64)
    labels = np.ascontiguousarray(get_col(['label_id', 'label'], 3), dtype=np.int64)

    labels_grid, x_min, vx_min, dx, dvx, num_x, num_vx = _build_grid(x, vx, labels)

    grid_scale = max(dx, dvx)
    if epsilons is None:
        lo = 2.5 * grid_scale
        hi = 0.4
        epsilons = np.logspace(np.log10(0.005), np.log10(0.05), 8)
    epsilons = np.asarray(epsilons, dtype=np.float64)

    phys_mask = np.isin(labels, np.array(PHYSICAL_LABELS, dtype=np.int64))
    phys_idx = np.nonzero(phys_mask)[0]
    n_phys = phys_idx.shape[0]
    sample_size = min(num_samples, n_phys)

    rng = np.random.default_rng(seed)

    idx = phys_idx[rng.choice(n_phys, size=sample_size, replace=False)]
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(epsilons.shape[0], sample_size))
    f_eps = _compute_uncertainty_numba(
        x, vx, labels, idx, epsilons, thetas,
        labels_grid, x_min, vx_min, dx, dvx, num_x, num_vx
    )
    fit_info = _fit_alpha(epsilons, f_eps)

    bootstrap_alphas = []
    for b in range(n_bootstrap):
        sub_rng = np.random.default_rng(seed + 10_000 + b)
        idx_b = phys_idx[sub_rng.choice(n_phys, size=sample_size, replace=True)]
        thetas_b = sub_rng.uniform(
            0.0, 2.0 * np.pi, size=(epsilons.shape[0], sample_size)
        )
        f_b = _compute_uncertainty_numba(
            x, vx, labels, idx_b, epsilons, thetas_b,
            labels_grid, x_min, vx_min, dx, dvx, num_x, num_vx
        )
        fit_b = _fit_alpha(epsilons, f_b)
        if not np.isnan(fit_b["alpha"]):
            bootstrap_alphas.append(fit_b["alpha"])

    bootstrap_alphas = np.array(bootstrap_alphas)

    if len(bootstrap_alphas) >= 10:
        ci_95 = (
            float(np.percentile(bootstrap_alphas, 2.5)),
            float(np.percentile(bootstrap_alphas, 97.5)),
        )
    else:
        ci_95 = (np.nan, np.nan)

    result = {
        "alpha": float(fit_info["alpha"]),
        "ci_95": ci_95,
        "r_squared": float(fit_info["r_squared"]),
        "reduced_chi2": float(fit_info["reduced_chi2"]),
        "stderr": float(fit_info["stderr_slope"]),
        "n_bootstrap": len(bootstrap_alphas),
        "bootstrap_alphas": bootstrap_alphas,
        "fit_info": fit_info,
    }

    extra = (epsilons, f_eps) if return_f else None
    return result, extra


def _save_pickle(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fig, f)


def plot_alpha_fit(epsilons, f_eps, result, C,
                   out_dir=FIGURE_DIR, pickle_dir=PICKLE_DIR):
    alpha = result["alpha"]
    ci = result["ci_95"]
    r2 = result["r_squared"]
    chi2 = result["reduced_chi2"]
    n_boot = result["n_bootstrap"]

    valid = f_eps > 0
    if valid.sum() < 2:
        print(f"C={C:.2f}: insufficient valid points for plotting")
        return None, None

    log_eps = np.log10(epsilons[valid])
    log_f = np.log10(f_eps[valid])

    plot_df = pd.DataFrame({"log_eps": log_eps, "log_f": log_f})

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    sns.regplot(
        data=plot_df, x="log_eps", y="log_f",
        scatter_kws={"s": 30, "color": "tab:blue", "zorder": 3},
        line_kws={"color": "crimson", "linestyle": "--", "linewidth": 1.8},
        ci=None, ax=ax,
    )

    stats_text = (
        f"$\\alpha = {alpha:.4f}$\n"
        f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
        f"$R^2 = {r2:.4f}$\n"
        f"$\\chi^2_\\nu = {chi2:.4f}$\n"
        f"$n_{{boot}} = {n_boot}$"
    )
    ax.text(
        0.05, 0.95, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.9),
    )

    ax.set_xlabel(r"$\log_{10}\epsilon$", fontsize=13)
    ax.set_ylabel(r"$\log_{10} f(\epsilon)$", fontsize=13)
    ax.set_title(f"Classical Uncertainty Exponent, $C={C:.2f}$", fontsize=14)

    fig.tight_layout()

    png_path = os.path.join(out_dir, f"alpha_fit_C{C:.2f}.png")
    pkl_path = os.path.join(pickle_dir, f"alpha_fit_C{C:.2f}.pkl")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    _save_pickle(fig, pkl_path)
    plt.close(fig)

    return png_path, pkl_path


def print_results_table(results):
    header = (
        f"{'C':>5} | {'α':>8} | {'95% CI':>20} | "
        f"{'R²':>7} | {'χ²ν':>7} | {'n_boot':>6}"
    )
    print("\n" + "=" * len(header))
    print("CLASSICAL UNCERTAINTY EXPONENT SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for C, res in sorted(results.items()):
        alpha = res["alpha"]
        ci = res["ci_95"]
        r2 = res["r_squared"]
        chi2 = res["reduced_chi2"]
        nb = res["n_bootstrap"]

        if np.isnan(alpha):
            print(f"{C:5.2f} | {'n/a':>8} | {'n/a':>20} | "
                  f"{'n/a':>7} | {'n/a':>7} | {nb:>6}")
        else:
            print(
                f"{C:5.2f} | {alpha:8.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{r2:7.4f} | {chi2:7.4f} | {nb:>6}"
            )

    print("=" * len(header))


if __name__ == "__main__":
    C_values = [3.00, 3.05, 3.10, 3.14, 3.20]
    N_BOOTSTRAP = 1000
    NUM_SAMPLES = 10000

    all_results = {}

    for C in C_values:
        path = f"files/data/{C:.2f}/700x700.csv"
        print(f"\nComputing α for C={C:.2f} ...")

        result, extra = compute_physical_alpha(
            path,
            num_samples=NUM_SAMPLES,
            n_bootstrap=N_BOOTSTRAP,
            seed=42,
            return_f=True,
        )

        all_results[C] = result

        if extra is not None and not np.isnan(result["alpha"]):
            epsilons, f_eps = extra
            png_path, pkl_path = plot_alpha_fit(epsilons, f_eps, result, C)
            if png_path:
                print(f"  PNG: {png_path}")
            if pkl_path:
                print(f"  PKL: {pkl_path}")

    print_results_table(all_results)

    out_rows = []
    for C, res in sorted(all_results.items()):
        out_rows.append({
            "C": C,
            "alpha": res["alpha"],
            "ci_lower": res["ci_95"][0],
            "ci_upper": res["ci_95"][1],
            "r_squared": res["r_squared"],
            "reduced_chi2": res["reduced_chi2"],
            "n_bootstrap": res["n_bootstrap"],
        })

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(FIGURE_DIR, "alpha_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved tabular results: {out_path}")