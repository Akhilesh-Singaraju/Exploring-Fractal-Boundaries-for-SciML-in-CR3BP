import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CR3BP_Neural_Fractals.postprocessing.unc_map import BasinClassifier, load_ensemble, DEVICE

DATA_DIR = "files/data"
PLOT_DIR = "files/figures"

ALPHA = {
    3.00: (0.1058, 0.0818, 0.1333),
    3.05: (0.1397, 0.0976, 0.1526),
    3.10: (0.1486, 0.1265, 0.1876),
    3.14: (0.1722, 0.1410, 0.2127),
}

def compute_direct_betas(C, num_samples=2000, n_dirs=8, n_bootstrap=100, seed=42):
    models, mean, scale, classes = load_ensemble(C)
    models = [m.to(DEVICE) for m in models]
    
    unc = pd.read_csv(os.path.join(DATA_DIR, f"uncertainty_C{C:.2f}.csv"))
    N = len(unc)
    n = min(num_samples, N)
    
    x_base = unc["x0"].to_numpy(dtype=np.float64)
    vx_base = unc["vx0"].to_numpy(dtype=np.float64)
    
    X_base_scaled = torch.tensor((np.column_stack([x_base, vx_base]) - mean) / scale, 
                                  dtype=torch.float32)
    
    base_probs = np.empty((N, len(models), len(classes)), dtype=np.float32)
    with torch.no_grad():
        batch_size = 4096
        for s in range(0, N, batch_size):
            b = X_base_scaled[s:s+batch_size].to(DEVICE)
            for mi, m in enumerate(models):
                base_probs[s:s+batch_size, mi, :] = F.softmax(m(b), dim=1).cpu().numpy()
    
    base_mean_probs = base_probs.mean(axis=1)  # (N, K)
    eps_const = 1e-12
    H_mean_base = -(base_mean_probs * np.log(base_mean_probs + eps_const)).sum(axis=1)
    H_ind_base = -(base_probs * np.log(base_probs + eps_const)).sum(axis=2)
    U_epi_base = H_mean_base - H_ind_base.mean(axis=1)
    pred_base = base_mean_probs.argmax(axis=1)
    
    grid_scale = 0.02
    epsilons = np.logspace(np.log10(2.5 * grid_scale), np.log10(0.4), 10)
    ne = len(epsilons)
    
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False)
    thetas = rng.uniform(0, 2*np.pi, size=(ne, n, n_dirs))  # (ne, n_samples, n_dirs)
    
    def evaluate_ensemble_at(x_phys, vx_phys):
        X_scaled = torch.tensor((np.column_stack([x_phys, vx_phys]) - mean) / scale, 
                                 dtype=torch.float32).to(DEVICE)
        probs_list = []
        with torch.no_grad():
            batch_size = 4096
            for s in range(0, len(X_scaled), batch_size):
                b = X_scaled[s:s+batch_size]
                batch_probs = []
                for m in models:
                    batch_probs.append(F.softmax(m(b), dim=1).cpu().numpy())
                probs_list.append(np.stack(batch_probs, axis=1))  # (batch, M, K)
        return np.concatenate(probs_list, axis=0)
    
    def compute_observables(idx_sub, thetas_sub):
        n_sub = len(idx_sub)
        flip = np.zeros(ne)
        dP = np.zeros(ne)
        dU = np.zeros(ne)
        n_valid = np.zeros(ne)
        
        for e_idx in range(ne):
            eps = epsilons[e_idx]
            f_sum, p_sum, u_sum, v_count = 0.0, 0.0, 0.0, 0.0
            
            chunk_size = 500
            for start in range(0, n_sub, chunk_size):
                end = min(start + chunk_size, n_sub)
                chunk_idx = idx_sub[start:end]
                chunk_thetas = thetas_sub[e_idx, start:end, :]  # (chunk, n_dirs)
                
                x_pert = x_base[chunk_idx][:, None] + eps * np.cos(chunk_thetas)
                vx_pert = vx_base[chunk_idx][:, None] + eps * np.sin(chunk_thetas)
                
                x_flat = x_pert.ravel()
                vx_flat = vx_pert.ravel()
                
                in_bounds = ((x_flat >= -1.5) & (x_flat <= 1.5) & 
                             (vx_flat >= -3.0) & (vx_flat <= 3.0))
                
                if not in_bounds.any():
                    continue
                
                x_eval = x_flat[in_bounds]
                vx_eval = vx_flat[in_bounds]
                pert_probs = evaluate_ensemble_at(x_eval, vx_eval)  # (n_eval, M, K)
                
                eval_ptr = 0
                for local_i, global_i in enumerate(chunk_idx):
                    for d in range(n_dirs):
                        flat_idx = local_i * n_dirs + d
                        if not in_bounds[flat_idx]:
                            continue
                        
                        p_k = pert_probs[eval_ptr]  # (M, K)
                        p_mean = p_k.mean(axis=0)
                        H_m = -(p_mean * np.log(p_mean + eps_const)).sum()
                        H_i = -(p_k * np.log(p_k + eps_const)).sum(axis=1).mean()
                        U_pert = H_m - H_i
                        pred_pert = p_mean.argmax()
                        
                        if pred_pert != pred_base[global_i]:
                            f_sum += 1.0
                        p_sum += np.abs(p_mean - base_mean_probs[global_i]).sum()
                        u_sum += abs(U_pert - U_epi_base[global_i])
                        v_count += 1.0
                        
                        eval_ptr += 1
            
            flip[e_idx] = f_sum / v_count if v_count > 0 else 0.0
            dP[e_idx] = p_sum / v_count if v_count > 0 else 0.0
            dU[e_idx] = u_sum / v_count if v_count > 0 else 0.0
            n_valid[e_idx] = v_count / (n_sub * n_dirs)
        
        return flip, dP, dU, n_valid
    
    flip, dP, dU, valid_frac = compute_observables(idx, thetas)
    
    def fit_window(obs, e_lo, e_hi):
        mask = (epsilons >= e_lo) & (epsilons <= e_hi) & (obs > 0)
        if mask.sum() < 3:
            return np.nan, np.nan
        log_e = np.log(epsilons[mask])
        if log_e.max() - log_e.min() < 0.3:
            return np.nan, np.nan
        r = linregress(log_e, np.log(obs[mask]))
        return r.slope, r.rvalue ** 2
    
    beta_flip_full, r2_flip_full = fit_window(flip, epsilons[0], epsilons[-1])
    beta_dP_full, r2_dP_full = fit_window(dP, epsilons[0], epsilons[-1])
    beta_dU_full, r2_dU_full = fit_window(dU, epsilons[0], epsilons[-1])
    
    EPS_MAX_FIT = 0.10
    beta_flip, r2_flip = fit_window(flip, epsilons[0], EPS_MAX_FIT)
    beta_dP, r2_dP = fit_window(dP, epsilons[0], EPS_MAX_FIT)
    beta_dU, r2_dU = fit_window(dU, epsilons[0], EPS_MAX_FIT)
    
    stability = {"flip": [], "dP": [], "dU": []}
    for i in range(len(epsilons) - 2):
        for j in range(i + 2, len(epsilons)):
            e_lo, e_hi = epsilons[i], epsilons[j]
            bf, _ = fit_window(flip, e_lo, e_hi)
            bp, _ = fit_window(dP, e_lo, e_hi)
            bu, _ = fit_window(dU, e_lo, e_hi)
            if not np.isnan(bf): stability["flip"].append((e_lo, e_hi, bf))
            if not np.isnan(bp): stability["dP"].append((e_lo, e_hi, bp))
            if not np.isnan(bu): stability["dU"].append((e_lo, e_hi, bu))
    
    boot_results = {"flip": [], "dP": [], "dU": []}
    for b in range(n_bootstrap):
        rb = np.random.default_rng(seed + 10000 + b)
        idx_b = rb.choice(N, size=n, replace=True)
        thetas_b = rb.uniform(0, 2*np.pi, size=(ne, n, n_dirs))
        f_b, p_b, u_b, _ = compute_observables(idx_b, thetas_b)
        for key, obs in (("flip", f_b), ("dP", p_b), ("dU", u_b)):
            slope, _ = fit_window(obs, epsilons[0], EPS_MAX_FIT)
            if not np.isnan(slope):
                boot_results[key].append(slope)
    
    def ci(key):
        v = boot_results[key]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) >= 10 else (np.nan, np.nan)
    
    return {
        "epsilons": epsilons,
        "flip": flip, "dP": dP, "dU": dU, "valid_frac": valid_frac,
        "beta_flip": (beta_flip, ci("flip"), r2_flip),
        "beta_dP": (beta_dP, ci("dP"), r2_dP),
        "beta_dU": (beta_dU, ci("dU"), r2_dU),
        "beta_flip_full": beta_flip_full,
        "r2_flip_full": r2_flip_full,
        "beta_dP_full": beta_dP_full,
        "r2_dP_full": r2_dP_full,
        "beta_dU_full": beta_dU_full,
        "r2_dU_full": r2_dU_full,
        "stability": stability,
    }
def plot_results(C, res):
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=200)
    for ax, (key, obs, lab) in zip(axes[:3], [
        ("flip", res["flip"], r"$f_{NN}(\epsilon)$"),
        ("dP", res["dP"], r"$\Delta P(\epsilon)$"),
        ("dU", res["dU"], r"$\Delta U_{epi}(\epsilon)$"),
    ]):
        ok = obs > 0
        ax.loglog(res["epsilons"][ok], obs[ok], "o-", label="data")
        b, _, _ = res[f"beta_{key}"]
        if not np.isnan(b):
            xx = res["epsilons"][ok]
            yy = obs[ok][0] * (xx / xx[0]) ** b
            ax.loglog(xx, yy, "r--", label=f"slope={b:.3f}")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    
    axes[3].semilogx(res["epsilons"], res["valid_frac"], "s-")
    axes[3].set_xlabel(r"$\epsilon$")
    axes[3].set_ylabel(r"$N_{valid}/N_{attempted}$")
    axes[3].set_ylim(0, 1.05)
    axes[3].grid(alpha=0.3)
    axes[3].set_title("Valid fraction")
    
    fig.suptitle(f"Direct-evaluation neural scaling, C={C:.2f}")
    path1 = os.path.join(PLOT_DIR, f"beta_direct_C{C:.2f}.png")
    fig.savefig(path1, bbox_inches="tight", dpi=200)
    plt.close(fig)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    for ax, (key, lab) in zip(axes, [("flip", r"$\beta_{flip}$"), ("dP", r"$\beta_{\Delta P}$"), ("dU", r"$\beta_{\Delta U}$")]):
        data = res["stability"][key]
        if data:
            lows, highs, betas = zip(*data)
            sc = ax.scatter(lows, highs, c=betas, cmap="viridis", s=20)
            plt.colorbar(sc, ax=ax, label=lab)
        ax.set_xlabel(r"$\epsilon_{min}$")
        ax.set_ylabel(r"$\epsilon_{max}$")
        ax.set_title(f"Stability: {lab}")
        ax.grid(alpha=0.3)
    
    path2 = os.path.join(PLOT_DIR, f"beta_stability_C{C:.2f}.png")
    fig.savefig(path2, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {path1}, {path2}")

if __name__ == "__main__":
    rows = []
    for C in [3.00, 3.05, 3.10, 3.14]:
        print(f"\nComputing direct betas for C={C:.2f}...")
        res = compute_direct_betas(C, num_samples=1500, n_dirs=8, n_bootstrap=50)
        plot_results(C, res)
        
        a, a_lo, a_hi = ALPHA[C]
        bf, bf_ci, r2f = res["beta_flip"]
        bp, bp_ci, r2p = res["beta_dP"]
        bu, bu_ci, r2u = res["beta_dU"]
        
        rows.append(dict(C=C, alpha=a, alpha_lo=a_lo, alpha_hi=a_hi,
                         beta_flip=bf, beta_flip_lo=bf_ci[0], beta_flip_hi=bf_ci[1], r2_flip=r2f,
                         beta_flip_full=res['beta_flip_full'], r2_flip_full=res['r2_flip_full'],
                         beta_dP=bp, beta_dP_lo=bp_ci[0], beta_dP_hi=bp_ci[1], r2_dP=r2p,
                         beta_dP_full=res['beta_dP_full'], r2_dP_full=res['r2_dP_full'],
                         beta_dU=bu, beta_dU_lo=bu_ci[0], beta_dU_hi=bu_ci[1], r2_dU=r2u,
                         beta_dU_full=res['beta_dU_full'], r2_dU_full=res['r2_dU_full']))
        
        print(f"C={C:.2f}  alpha={a:.4f} [{a_lo:.4f},{a_hi:.4f}]")
        print(f"   beta_flip (restricted)={bf:.4f} [{bf_ci[0]:.4f},{bf_ci[1]:.4f}]  R2={r2f:.3f} | (full)={res['beta_flip_full']:.4f}")
        print(f"   beta_dP   (restricted)={bp:.4f} [{bp_ci[0]:.4f},{bp_ci[1]:.4f}]  R2={r2p:.3f} | (full)={res['beta_dP_full']:.4f}")
        print(f"   beta_dU   (restricted)={bu:.4f} [{bu_ci[0]:.4f},{bu_ci[1]:.4f}]  R2={r2u:.3f} | (full)={res['beta_dU_full']:.4f}")
        print(f"   Valid frac range: [{res['valid_frac'].min():.3f}, {res['valid_frac'].max():.3f}]")
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "alpha_beta_direct_comparison.csv"), index=False)
    print("\nSaved comparison to files/data/alpha_beta_direct_comparison.csv")