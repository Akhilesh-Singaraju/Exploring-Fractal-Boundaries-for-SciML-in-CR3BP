import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

df = pd.read_csv("files/sweep_results/sweep_summary.csv")

agg = df.groupby(["ff_scale", "hidden"]).agg({
    "val_ba": "mean",
    "D_box": "mean"
}).reset_index()

os.makedirs("files/sweep_results/figures", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(8, 6))
for scale in sorted(agg["ff_scale"].unique()):
    sub = agg[agg["ff_scale"] == scale]
    ax1.plot(sub["hidden"], sub["D_box"], marker="o", linewidth=2, label=f"Scale {scale}")

ax1.set_xscale("log")
ax1.set_xticks([256, 512, 1024])
ax1.set_xticklabels(["256", "512", "1024"])
ax1.set_xlabel("Hidden Dimension (Network Capacity)")
ax1.set_ylabel("Learned Boundary Dimension ($D_{box}$)")
ax1.set_title("Capacity Independence of Geometric Collapse")
ax1.legend(title="Fourier Scale")
ax1.grid(True, which="both", ls="--", alpha=0.7)
fig1.tight_layout()

fig1.savefig("files/sweep_results/figures/capacity_independence.png", dpi=300)
with open("files/sweep_results/figures/capacity_independence.pkl", "wb") as f:
    pickle.dump(fig1, f)
plt.close(fig1)

scale_agg = df.groupby("ff_scale").agg({
    "val_ba": "mean",
    "D_box": "mean"
}).reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2_twin = ax2.twinx()

bars = ax2.bar(scale_agg["ff_scale"].astype(str), scale_agg["D_box"], color="tab:blue", alpha=0.7, label="$D_{box}$ (Geometry)")
line = ax2_twin.plot(scale_agg["ff_scale"].astype(str), scale_agg["val_ba"], color="tab:red", marker="s", linewidth=2.5, markersize=8, label="Val Balanced Accuracy (Topology)")

ax2.set_xlabel("Fourier Frequency Scale (Bandwidth)")
ax2.set_ylabel("Learned Boundary Dimension ($D_{box}$)", color="tab:blue")
ax2_twin.set_ylabel("Validation Balanced Accuracy", color="tab:red")

ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2_twin.tick_params(axis="y", labelcolor="tab:red")

ax2.set_ylim(0.8, 1.3)
ax2_twin.set_ylim(0.6, 0.85)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right")

ax2.set_title("The Spectral Trade-off: Geometry vs. Topology")
ax2.grid(True, axis="y", ls="--", alpha=0.5)
fig2.tight_layout()

fig2.savefig("files/sweep_results/figures/spectral_tradeoff.png", dpi=300)
with open("files/sweep_results/figures/spectral_tradeoff.pkl", "wb") as f:
    pickle.dump(fig2, f)
plt.close(fig2)

print("Plots and pickles saved to files/sweep_results/figures/")