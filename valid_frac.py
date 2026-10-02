import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from beta import compute_direct_betas, ALPHA
except ImportError:
    try:
        from beta_analysis import compute_neural_betas as compute_direct_betas, ALPHA
    except ImportError:
        print("Error: Could not import beta computation function.")
        print("Ensure beta_analysis_direct.py or beta_analysis.py is in the path.")
        exit(1)

DATA_DIR = "files/data"
PLOT_DIR = "files/figures"

def analyze_valid_fraction(C, num_samples=2000, n_dirs=8, seed=42):
    print(f"\n--- Analyzing Valid Fraction for C={C:.2f} ---")
    
    res = compute_direct_betas(C, num_samples=num_samples, n_dirs=n_dirs, n_bootstrap=0, seed=seed)
    
    epsilons = res["epsilons"]
    valid_frac = res["valid_frac"]
    obs_dU = res["dU"]
    
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=200)
    
    color = 'tab:blue'
    ax1.set_xlabel(r'$\epsilon$')
    ax1.set_ylabel(r'Valid Fraction $q(\epsilon)$', color=color)
    ax1.semilogx(epsilons, valid_frac, 'o-', color=color, label=r'$q(\epsilon)$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3, which='both')
    ax1.axhline(0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.legend(loc='lower left')
    
    betas = []
    r2s = []
    max_eps_used = []
    
    min_points_for_fit = 4
    
    for k in range(len(epsilons), min_points_for_fit - 1, -1):
        e_subset = epsilons[:k]
        o_subset = obs_dU[:k]
        
        ok = o_subset > 0
        if ok.sum() < 3:
            continue
            
        log_e = np.log(e_subset[ok])
        log_o = np.log(o_subset[ok])
        
        if log_e.max() - log_e.min() < 0.3:
            continue
            
        slope, _, r, _, _ = linregress(log_e, log_o)
        betas.append(slope)
        r2s.append(r**2)
        max_eps_used.append(e_subset[-1])
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'$\beta_{\Delta U}$ (cumulative fit)', color=color)
    ax2.semilogx(max_eps_used, betas, 's-', color=color, label=r'$\beta$ fit')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title(f"Valid Fraction & Beta Stability (C={C:.2f})")
    fig.tight_layout()
    
    out_path = os.path.join(PLOT_DIR, f"valid_fraction_diag_C{C:.2f}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved diagnostic plot: {out_path}")
    
    print(f"  Epsilon range: [{epsilons[0]:.4f}, {epsilons[-1]:.4f}]")
    print(f"  Valid fraction range: [{valid_frac.min():.3f}, {valid_frac.max():.3f}]")
    
    if len(betas) > 0:
        beta_full = betas[0]
        beta_restricted = betas[-1]
        
        print(f"  Beta (full range):      {beta_full:.4f} (R2={r2s[0]:.3f})")
        print(f"  Beta (restricted range): {beta_restricted:.4f} (R2={r2s[-1]:.3f})")
        print(f"  Max epsilon in restricted fit: {max_eps_used[-1]:.4f}")
        
        diff = abs(beta_full - beta_restricted)
        print(f"    Difference:{diff}")
    else:
        print("  Could not perform stability fit (insufficient points).")

if __name__ == "__main__":
    C_VALUES = [3.00, 3.05, 3.10, 3.14]
    for C in C_VALUES:
        analyze_valid_fraction(C)