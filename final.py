import os
import pandas as pd
from integrate import generate_basin_grid

C_VALUES = [3.00, 3.05, 3.10, 3.14, 3.20]
T_MAX = 12000.0
GRID_RESOLUTION = 300

os.makedirs("files/data", exist_ok=True)

for C in C_VALUES:
    print(f"--- Generating dataset for C = {C} (T = {T_MAX}) ---")
    df = generate_basin_grid(
        C=C, 
        t_max=T_MAX, 
        res_x=GRID_RESOLUTION, 
        res_vx=GRID_RESOLUTION
    )
    output_path = f"files/basin_C{C:.2f}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")