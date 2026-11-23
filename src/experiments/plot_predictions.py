from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results" / "predictions" / "iid_predictions.csv"
OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "prediction_vs_actual_iid.png"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading predictions...")

df = pd.read_csv(INPUT)

y = df["Actual_Power_W"]

models = {
    "Physics": df["Physics_Prediction_W"],
    "Pure ML": df["PureML_Prediction_W"],
    "Hybrid Residual": df["Hybrid_Prediction_W"],
}


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

all_values = pd.concat(
    [df["Actual_Power_W"],
     df["Physics_Prediction_W"],
     df["PureML_Prediction_W"],
     df["Hybrid_Prediction_W"]]
)

axis_min = all_values.min()
axis_max = all_values.max()

margin = 0.03 * (axis_max - axis_min)

axis_min -= margin
axis_max += margin


for ax, (name, prediction) in zip(axes, models.items()):

    rmse = np.sqrt(mean_squared_error(y, prediction))
    r2 = r2_score(y, prediction)

    ax.scatter(
        y,
        prediction,
        alpha=0.7,
        s=35
    )

    # Perfect prediction line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title(name)

    ax.text(
        0.05,
        0.92,
        f"RMSE = {rmse:.3f} W\nR² = {r2:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.8)
    )

    ax.grid(alpha=0.25)


fig.suptitle(
    "Actual vs Predicted PV Power — IID Test Set",
    fontsize=15
)

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