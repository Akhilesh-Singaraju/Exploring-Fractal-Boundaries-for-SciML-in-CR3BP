import numpy as np
import pandas as pd


for c in [3.00, 3.05, 3.10, 3.14, 3.20]:
    df = pd.read_csv(f"files/data(ARCHIVE)/basin_C{c:.2f}.csv")
    valid = df[df["outcome"] != "INVALID_ENERGY"]
    pivot = valid.pivot_table(values="label_id", index="vx0", columns="x0", aggfunc="first")
    grid = pivot.values
    dy, dx = np.gradient(grid.astype(float))
    bf = ((dx != 0) | (dy != 0)).sum() / grid.size
    print(f"C={c:.2f}: boundary_fraction={bf:.4f}, valid_points={len(valid)}")