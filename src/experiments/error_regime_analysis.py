from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results" / "predictions" / "iid_predictions.csv"

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading prediction data...")

df = pd.read_csv(INPUT)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CALCULATE ABSOLUTE ERRORS
# ============================================================

df["Physics_Abs_Error_W"] = df["Physics_Error_W"].abs()
df["PureML_Abs_Error_W"] = df["PureML_Error_W"].abs()
df["Hybrid_Abs_Error_W"] = df["Hybrid_Error_W"].abs()


# ============================================================
# 1. ERROR DISTRIBUTION
# ============================================================

print()
print("Creating error distribution plot...")

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    df["Physics_Abs_Error_W"],
    bins=20,
    alpha=0.55,
    label="Physics"
)

ax.hist(
    df["PureML_Abs_Error_W"],
    bins=20,
    alpha=0.55,
    label="Pure ML"
)

ax.hist(
    df["Hybrid_Abs_Error_W"],
    bins=20,
    alpha=0.55,
    label="Hybrid Residual"
)

ax.set_xlabel("Absolute Error (W)")
ax.set_ylabel("Number of Test Samples")
ax.set_title("Absolute Prediction Error Distribution — IID Test Set")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_distribution_iid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. ERROR VS IRRADIANCE
# ============================================================

print("Creating error vs irradiance plot...")

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    df["irradiance_wm2"],
    df["Physics_Abs_Error_W"],
    alpha=0.6,
    label="Physics"
)

ax.scatter(
    df["irradiance_wm2"],
    df["PureML_Abs_Error_W"],
    alpha=0.6,
    label="Pure ML"
)

ax.scatter(
    df["irradiance_wm2"],
    df["Hybrid_Abs_Error_W"],
    alpha=0.6,
    label="Hybrid Residual"
)

ax.set_xlabel("Irradiance (W/m²)")
ax.set_ylabel("Absolute Error (W)")
ax.set_title("Prediction Error vs Irradiance — IID Test Set")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_vs_irradiance_iid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. ERROR VS TEMPERATURE
# ============================================================

print("Creating error vs temperature plot...")

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    df["temperature_c"],
    df["Physics_Abs_Error_W"],
    alpha=0.6,
    label="Physics"
)

ax.scatter(
    df["temperature_c"],
    df["PureML_Abs_Error_W"],
    alpha=0.6,
    label="Pure ML"
)

ax.scatter(
    df["temperature_c"],
    df["Hybrid_Abs_Error_W"],
    alpha=0.6,
    label="Hybrid Residual"
)

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Absolute Error (W)")
ax.set_title("Prediction Error vs Temperature — IID Test Set")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_vs_temperature_iid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ERROR VS SHADING
# ============================================================

print("Creating error vs shading plot...")

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    df["shading_fraction"] * 100,
    df["Physics_Abs_Error_W"],
    alpha=0.6,
    label="Physics"
)

ax.scatter(
    df["shading_fraction"] * 100,
    df["PureML_Abs_Error_W"],
    alpha=0.6,
    label="Pure ML"
)

ax.scatter(
    df["shading_fraction"] * 100,
    df["Hybrid_Abs_Error_W"],
    alpha=0.6,
    label="Hybrid Residual"
)

ax.set_xlabel("Shading Fraction (%)")
ax.set_ylabel("Absolute Error (W)")
ax.set_title("Prediction Error vs Shading — IID Test Set")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_vs_shading_iid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. SUMMARY BY SHADING LEVEL
# ============================================================

print()
print("========================================")
print("ERROR SUMMARY BY SHADING")
print("========================================")

shading_summary = (
    df.groupby("shading_fraction")
    .agg(
        Physics_MAE=("Physics_Abs_Error_W", "mean"),
        PureML_MAE=("PureML_Abs_Error_W", "mean"),
        Hybrid_MAE=("Hybrid_Abs_Error_W", "mean"),
        Physics_MaxError=("Physics_Abs_Error_W", "max"),
        PureML_MaxError=("PureML_Abs_Error_W", "max"),
        Hybrid_MaxError=("Hybrid_Abs_Error_W", "max"),
    )
    .reset_index()
)

shading_summary["shading_pct"] = (
    shading_summary["shading_fraction"] * 100
)

print(
    shading_summary[
        [
            "shading_pct",
            "Physics_MAE",
            "PureML_MAE",
            "Hybrid_MAE",
            "Physics_MaxError",
            "PureML_MaxError",
            "Hybrid_MaxError",
        ]
    ].round(3).to_string(index=False)
)


# ============================================================
# 6. WORST CASES
# ============================================================

print()
print("========================================")
print("WORST HYBRID CASES")
print("========================================")

worst = (
    df.nlargest(10, "Hybrid_Abs_Error_W")
    [
        [
            "irradiance_wm2",
            "temperature_c",
            "shading_fraction",
            "Actual_Power_W",
            "Physics_Abs_Error_W",
            "PureML_Abs_Error_W",
            "Hybrid_Abs_Error_W",
        ]
    ]
)

print(worst.round(3).to_string(index=False))


# ============================================================
# SAVE ERROR-ENRICHED DATA
# ============================================================

OUTPUT_CSV = (
    ROOT / "results" / "metrics" / "iid_error_analysis.csv"
)

df.to_csv(OUTPUT_CSV, index=False)


print()
print("========================================")
print("ERROR ANALYSIS COMPLETE")
print("========================================")
print(f"Figures saved to: {OUTPUT_DIR}")
print(f"Detailed data saved to: {OUTPUT_CSV}")