from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results" / "metrics" / "final_model_comparison.csv"
OUTPUT = ROOT / "results" / "figures" / "hybrid_rmse_improvement.png"


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT)

# Calculate percentage RMSE reduction of Hybrid vs Physics
df["Hybrid_vs_Physics_%"] = (
    (df["Physics_RMSE"] - df["Hybrid_RMSE"])
    / df["Physics_RMSE"]
    * 100
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    df["experiment"],
    df["Hybrid_vs_Physics_%"]
)

ax.set_ylabel("RMSE reduction vs Physics (%)")
ax.set_xlabel("Evaluation regime")
ax.set_title(
    "Hybrid Residual Model: RMSE Reduction vs Physics Baseline"
)

ax.set_ylim(0, 105)

ax.grid(
    axis="y",
    alpha=0.25
)

# Add values above bars
for bar, value in zip(bars, df["Hybrid_vs_Physics_%"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("========================================")
print("PLOT SAVED")
print("========================================")
print(f"Saved to: {OUTPUT}")