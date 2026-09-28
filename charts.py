import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data_dir = "files"
c_values = [3.00, 3.05, 3.10, 3.14, 3.20]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
cmap = ListedColormap(colors[:4])

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=300)

for idx, C in enumerate(c_values):
    file_path = f"{data_dir}/basin_C{C:.2f}.csv"
    if not os.path.exists(file_path):
        print(f"File missing: {file_path}")
        continue
        
    df = pd.read_csv(file_path)
    
    x_col = [c for c in df.columns if 'x' in c.lower() and 'v' not in c.lower()][0]
    vx_col = [c for c in df.columns if 'v' in c.lower()][0]
    label_col = [c for c in df.columns if 'label' in c.lower() or 'state' in c.lower() or 'target' in c.lower()][0]

    grid = df.pivot(index=vx_col, columns=x_col, values=label_col)
    
    ax = axes[idx]
    im = ax.imshow(
        grid.values, 
        extent=[df[x_col].min(), df[x_col].max(), df[vx_col].min(), df[vx_col].max()],
        origin='lower', 
        aspect='auto', 
        cmap=cmap
    )
    
    ax.set_title(f"Jacobi Constant C = {C:.2f}", fontsize=12, fontweight='bold')
    ax.set_xlabel("\(x\) [LU]", fontsize=10)
    if idx == 0:
        ax.set_ylabel("\(v_x\) [VU]", fontsize=10)
    else:
        ax.set_yticklabels([])

plt.tight_layout()
os.makedirs("files/plots", exist_ok=True)
plt.savefig("files/plots/basin_comparison_all_C.png", bbox_inches='tight', dpi=300)
plt.show()
print("Basin visualization saved to: files/plots/basin_comparison_all_C.png")