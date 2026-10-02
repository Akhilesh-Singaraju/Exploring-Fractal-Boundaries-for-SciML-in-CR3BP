import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from numba import njit, prange

PHYSICAL_LABELS = (1, 2, 3)   # Earth, Moon, Escape

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
    if valid.sum() < 2:
        return np.nan, np.nan
    res = linregress(np.log(epsilons[valid]), np.log(f_eps[valid]))
    return res.slope, res.rvalue ** 2

def compute_physical_alpha(csv_path, num_samples=3000, epsilons=None,
                           n_bootstrap=200, seed=42, return_f=False):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return np.nan, (np.nan, np.nan), None

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
        epsilons = np.logspace(np.log10(lo), np.log10(hi), 10)
    epsilons = np.asarray(epsilons, dtype=np.float64)

    phys_mask = np.isin(labels, np.array(PHYSICAL_LABELS, dtype=np.int64))
    phys_idx = np.nonzero(phys_mask)[0]
    n_phys = phys_idx.shape[0]
    sample_size = min(num_samples, n_phys)

    rng = np.random.default_rng(seed)

    def _one_draw(sub_rng):
        idx = phys_idx[sub_rng.choice(n_phys, size=sample_size, replace=False)]
        thetas = sub_rng.uniform(0.0, 2.0 * np.pi,
                                 size=(epsilons.shape[0], sample_size))
        return _compute_uncertainty_numba(x, vx, labels, idx, epsilons, thetas,
                                          labels_grid, x_min, vx_min, dx, dvx,
                                          num_x, num_vx)

    f_eps = _one_draw(rng)
    alpha, r2 = _fit_alpha(epsilons, f_eps)

    alphas = []
    for b in range(n_bootstrap):
        sub_rng = np.random.default_rng(seed + 10_000 + b)
        idx_b = phys_idx[sub_rng.choice(n_phys, size=sample_size, replace=True)]
        thetas_b = sub_rng.uniform(0.0, 2.0 * np.pi,
                                   size=(epsilons.shape[0], sample_size))
        f_b = _compute_uncertainty_numba(x, vx, labels, idx_b, epsilons, thetas_b,
                                         labels_grid, x_min, vx_min, dx, dvx,
                                         num_x, num_vx)
        a_b, _ = _fit_alpha(epsilons, f_b)
        if not np.isnan(a_b):
            alphas.append(a_b)

    if len(alphas) >= 10:
        ci = (np.percentile(alphas, 2.5), np.percentile(alphas, 97.5))
    else:
        ci = (np.nan, np.nan)

    if return_f:
        return alpha, ci, (epsilons, f_eps, r2)
    return alpha, ci, r2


def plot_alpha_fit(epsilons, f_eps, alpha, ci, C, out_dir="files/figures"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    valid = f_eps > 0
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    ax.scatter(np.log10(epsilons[valid]), np.log10(f_eps[valid]),
               color='tab:blue', zorder=3)
    xe = np.log10(epsilons[valid])
    ye = alpha * xe + (np.log10(f_eps[valid][0]) - alpha * xe[0])
    ax.plot(xe, ye, 'r--', lw=1.5,
            label=r"$\alpha=%.3f$  [%.3f, %.3f]" % (alpha, ci[0], ci[1]))
    ax.set_xlabel(r"$\log_{10}\epsilon$")
    ax.set_ylabel(r"$\log_{10} f(\epsilon)$")
    ax.set_title(f"Uncertainty exponent, C={C:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, f"alpha_fit_C{C:.2f}.png")
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    C_values = [3.00, 3.05, 3.10, 3.14, 3.20]

    print(f"{'C':>5} | {'alpha':>7} | {'95% CI':>18} | {'R2':>6}")
    print("-" * 50)
    for C in C_values:
        path = f"files/data/basin_C{C:.2f}.csv"
        alpha, ci, extra = compute_physical_alpha(
            path, num_samples=3000, n_bootstrap=200, seed=42, return_f=True
        )
        if extra is not None:
            epsilons, f_eps, r2 = extra
            plot_alpha_fit(epsilons, f_eps, alpha, ci, C)
            print(f"{C:5.2f} | {alpha:7.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r2:6.4f}")
        else:
            print(f"{C:5.2f} | {'n/a':>7} | {'n/a':>18} | {'n/a':>6}")