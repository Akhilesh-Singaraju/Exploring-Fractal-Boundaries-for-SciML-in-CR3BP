# 1) Inspecting data
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "files/basin_C3.14.csv"

df = pd.read_csv(file)

print("\nRows:", len(df))
print("\nOutcome counts:")
print(df["outcome"].value_counts())

print("\nOutcome fractions:")
print(df["outcome"].value_counts(normalize=True))

print("\nMissing values:")
print(df.isna().sum())

print("\nJacobi constant:")
print(df["C"].unique())

print("\nEvent-time summary:")
print(df["t_event"].describe())

print("\nTrajectory diagnostics:")
print(df[["min_r_earth", "min_r_moon", "max_r", "n_steps"]].describe())

# 2) Ploting Basins

LABELS = {"IMPACT_EARTH": 1, "IMPACT_MOON": 2, "ESCAPE": 3, "MAX_TIME": 4, "INVALID_ENERGY": 0}


def plot_basin(csv_path, output_png):
    df = pd.read_csv(csv_path)
    xs = np.sort(df.x0.unique())
    vxs = np.sort(df.vx0.unique())
    Z = np.full((len(vxs), len(xs)), np.nan)

    x_to_i = {x: i for i, x in enumerate(xs)}
    vx_to_i = {v: i for i, v in enumerate(vxs)}
    for row in df.itertuples():
        Z[vx_to_i[row.vx0], x_to_i[row.x0]] = LABELS[row.outcome]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        origin="lower",
        extent=[xs.min(), xs.max(), vxs.min(), vxs.max()],
        interpolation="nearest",
        aspect="auto",
    )
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$\dot{x}_0$")
    plt.title(f"Earth–Moon CR3BP basin map, C={df.C.iloc[0]:.5f}")
    plt.colorbar(label="1=Earth, 2=Moon, 3=Escape, 4=Unresolved")
    plt.tight_layout()
    plt.savefig(output_png, dpi=250)
    plt.close()

plot_basin(file, "output.png")