import pandas as pd
from integrate import integrate_single_numba, OUTCOME_MAP

df = pd.read_csv("files/basin_C3.14.csv")

unresolved = df[df["outcome"] == "MAX_TIME"].sample(
    n=min(100, len(df[df["outcome"] == "MAX_TIME"])),
    random_state=42,
)

print(f"Testing {len(unresolved)} MAX_TIME trajectories")


for t_max in [100.0, 300.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0]:

    counts = {}

    for _, row in unresolved.iterrows():

        res = integrate_single_numba(
            row["x0"],
            row["vx0"],
            row["C"],
            t_max,
        )

        outcome = OUTCOME_MAP[res[0]]
        counts[outcome] = counts.get(outcome, 0) + 1

    print()
    print("=" * 50)
    print(f"t_max = {t_max}")
    print("=" * 50)

    total = sum(counts.values())

    for outcome, count in sorted(counts.items()):
        print(f"{outcome:20s} {count:4d}  ({count / total:.1%})")


sample = df[df["outcome"] == "MAX_TIME"].sample(
    n=min(100, len(df[df["outcome"] == "MAX_TIME"])),
    random_state=42,
)

print("Integrating sample to t_max = 12000...\n")

long_lived = []

for _, row in sample.iterrows():

    res = integrate_single_numba(
        row["x0"],
        row["vx0"],
        row["C"],
        12000.0,
    )

    outcome = OUTCOME_MAP[res[0]]

    if outcome == "MAX_TIME":
        long_lived.append({
            "x0": row["x0"],
            "vx0": row["vx0"],
            "outcome": outcome,
            "t_event": res[1],
            "n_steps": res[2],
            "min_r_earth": res[3],
            "min_r_moon": res[4],
            "max_r": res[5],
        })


result = pd.DataFrame(long_lived)

print(f"Remaining MAX_TIME trajectories: {len(result)}")
print()

if len(result) > 0:
    print(result.to_string(index=False))

    print("\nSummary:")
    print(result[[
        "min_r_earth",
        "min_r_moon",
        "max_r",
        "n_steps"
    ]].describe())