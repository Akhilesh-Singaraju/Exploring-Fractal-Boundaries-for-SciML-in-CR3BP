import numpy as np
import pandas as pd

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0

df = pd.read_csv("files/data/basin_C3.05.csv")

r1_0 = (df["x0"] + MU).abs()
r2_0 = (df["x0"] - 1.0 + MU).abs()

omega0 = 0.5 * df["x0"] ** 2 + (1.0 - MU) / r1_0 + MU / r2_0

vy2 = 2.0 * omega0 - df["C"] - df["vx0"] ** 2

expected_invalid = vy2 < 0
actual_invalid = df["label_id"] == 0

bad_energy = (expected_invalid != actual_invalid).sum()

print(f"INVALID_ENERGY classification errors: " f"{bad_energy} / {len(df)}")

print("t_event > t_max rows:", (df["t_event"] > 5000.0 + 1e-6).sum())

print("negative t_event rows:", (df["t_event"] < 0).sum())

print(
    "terminal rows with t_event=0:",
    ((df["label_id"].isin([1, 2, 3])) & (df["t_event"] <= 0)).sum(),
)

viol = {
    "IMPACT_EARTH rows with min_r_earth > R_EARTH": (
        (df["label_id"] == 1) & (df["min_r_earth"] > R_EARTH)
    ).sum(),
    "IMPACT_MOON rows with min_r_moon > R_MOON": (
        (df["label_id"] == 2) & (df["min_r_moon"] > R_MOON)
    ).sum(),
    "ESCAPE rows with max_r < R_ESCAPE": (
        (df["label_id"] == 3) & (df["max_r"] < R_ESCAPE)
    ).sum(),
    "MAX_TIME rows with max_r >= R_ESCAPE": (
        (df["label_id"] == 4) & (df["max_r"] >= R_ESCAPE)
    ).sum(),
    "MAX_TIME rows that graze Earth (min_r_earth <= R_EARTH)": (
        (df["label_id"] == 4) & (df["min_r_earth"] <= R_EARTH)
    ).sum(),
    "MAX_TIME rows that graze Moon (min_r_moon <= R_MOON)": (
        (df["label_id"] == 4) & (df["min_r_moon"] <= R_MOON)
    ).sum(),
}

for k, v in viol.items():
    print(f"{k}: {v}")

print(
    "\nn_steps: min/median/max =",
    df["n_steps"].min(),
    df["n_steps"].median(),
    df["n_steps"].max(),
)

print(
    "terminal rows with n_steps = 0:",
    ((df["label_id"].isin([1, 2, 3])) & (df["n_steps"] == 0)).sum(),
)

print("any NaNs:", df.isna().sum().sum())
