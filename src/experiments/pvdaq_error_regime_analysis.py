from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PRED_PATH = ROOT / "results" / "predictions" / "pvdaq_2025_predictions.csv"
OUT_METRICS = ROOT / "results" / "metrics" / "pvdaq_error_regime_analysis.csv"
OUT_DIR = ROOT / "results" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(PRED_PATH)
# Standardise measured power column name
if "Actual_Power_kW" not in df.columns:
    df = df.rename(
        columns={
            "Measured_DC_Power_kW": "Actual_Power_kW"
        }
    )

print("=" * 70)
print("PVDAQ ERROR-REGIME ANALYSIS")
print("=" * 70)

print(f"\nLoaded: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

# Expected columns from pvdaq validation:
# timestamp, poa_irradiance_wm2, ambient_temperature_c,
# wind_speed_ms, t_cell, Actual_Power_kW,
# Physics_Prediction_kW, PureML_Prediction_kW,
# Hybrid_Prediction_kW

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="ISO8601",
    errors="raise"
)

df["Physics_Error_kW"] = (
    df["Physics_Prediction_kW"] - df["Actual_Power_kW"]
)

df["PureML_Error_kW"] = (
    df["PureML_Prediction_kW"] - df["Actual_Power_kW"]
)

df["Hybrid_Error_kW"] = (
    df["Hybrid_Prediction_kW"] - df["Actual_Power_kW"]
)

df["Physics_AbsError_kW"] = df["Physics_Error_kW"].abs()
df["PureML_AbsError_kW"] = df["PureML_Error_kW"].abs()
df["Hybrid_AbsError_kW"] = df["Hybrid_Error_kW"].abs()


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\nOverall error summary:")

summary = pd.DataFrame({
    "Model": ["Physics", "Pure ML", "Hybrid Residual"],
    "MAE_kW": [
        df["Physics_AbsError_kW"].mean(),
        df["PureML_AbsError_kW"].mean(),
        df["Hybrid_AbsError_kW"].mean(),
    ],
    "RMSE_kW": [
        np.sqrt(np.mean(df["Physics_Error_kW"] ** 2)),
        np.sqrt(np.mean(df["PureML_Error_kW"] ** 2)),
        np.sqrt(np.mean(df["Hybrid_Error_kW"] ** 2)),
    ],
    "Bias_kW": [
        df["Physics_Error_kW"].mean(),
        df["PureML_Error_kW"].mean(),
        df["Hybrid_Error_kW"].mean(),
    ],
})

print(summary.round(4))


# ============================================================
# FUNCTION FOR REGIME ANALYSIS
# ============================================================

def regime_analysis(data, column, bins, labels):

    data = data.copy()

    data["regime"] = pd.cut(
        data[column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    rows = []

    for regime, group in data.groupby("regime", observed=False):

        if len(group) == 0:
            continue

        physics_rmse = np.sqrt(
            np.mean(group["Physics_Error_kW"] ** 2)
        )

        pureml_rmse = np.sqrt(
            np.mean(group["PureML_Error_kW"] ** 2)
        )

        hybrid_rmse = np.sqrt(
            np.mean(group["Hybrid_Error_kW"] ** 2)
        )

        improvement = (
            (physics_rmse - hybrid_rmse)
            / physics_rmse
            * 100
        )

        rows.append({
            "regime": str(regime),
            "n_samples": len(group),

            "Physics_MAE_kW":
                group["Physics_AbsError_kW"].mean(),

            "PureML_MAE_kW":
                group["PureML_AbsError_kW"].mean(),

            "Hybrid_MAE_kW":
                group["Hybrid_AbsError_kW"].mean(),

            "Physics_RMSE_kW":
                physics_rmse,

            "PureML_RMSE_kW":
                pureml_rmse,

            "Hybrid_RMSE_kW":
                hybrid_rmse,

            "Hybrid_vs_Physics_improvement_pct":
                improvement,
        })

    return pd.DataFrame(rows)


# ============================================================
# IRRADIANCE REGIMES
# ============================================================

irr_bins = [0, 100, 300, 500, 700, 900, 1100, np.inf]

irr_labels = [
    "0-100",
    "100-300",
    "300-500",
    "500-700",
    "700-900",
    "900-1100",
    ">1100"
]

irr_results = regime_analysis(
    df,
    "poa_irradiance_wm2",
    irr_bins,
    irr_labels
)

print("\n" + "=" * 70)
print("IRRADIANCE REGIME")
print("=" * 70)

print(
    irr_results.to_string(index=False, float_format=lambda x: f"{x:.3f}")
)


# ============================================================
# TEMPERATURE REGIMES
# ============================================================

temp_bins = [-np.inf, 10, 20, 30, 40, np.inf]

temp_labels = [
    "<=10",
    "10-20",
    "20-30",
    "30-40",
    ">40"
]

temp_results = regime_analysis(
    df,
    "t_cell",
    temp_bins,
    temp_labels
)

print("\n" + "=" * 70)
print("CELL TEMPERATURE REGIME")
print("=" * 70)

print(
    temp_results.to_string(index=False, float_format=lambda x: f"{x:.3f}")
)


# ============================================================
# WIND REGIMES
# ============================================================

wind_bins = [0, 1, 2, 3, 5, 8, np.inf]

wind_labels = [
    "0-1",
    "1-2",
    "2-3",
    "3-5",
    "5-8",
    ">8"
]

wind_results = regime_analysis(
    df,
    "wind_speed_ms",
    wind_bins,
    wind_labels
)

print("\n" + "=" * 70)
print("WIND REGIME")
print("=" * 70)

print(
    wind_results.to_string(index=False, float_format=lambda x: f"{x:.3f}")
)


# ============================================================
# HOUR OF DAY
# ============================================================

df["hour"] = df["timestamp"].dt.hour

hour_bins = [0, 8, 10, 12, 14, 16, 18, 24]

hour_labels = [
    "00-08",
    "08-10",
    "10-12",
    "12-14",
    "14-16",
    "16-18",
    "18-24"
]

hour_results = regime_analysis(
    df,
    "hour",
    hour_bins,
    hour_labels
)

print("\n" + "=" * 70)
print("TIME-OF-DAY REGIME")
print("=" * 70)

print(
    hour_results.to_string(index=False, float_format=lambda x: f"{x:.3f}")
)


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = []

for name, result in [
    ("irradiance", irr_results),
    ("temperature", temp_results),
    ("wind", wind_results),
    ("hour", hour_results),
]:

    temp = result.copy()
    temp.insert(0, "regime_type", name)
    all_results.append(temp)

final_results = pd.concat(
    all_results,
    ignore_index=True
)

final_results.to_csv(
    OUT_METRICS,
    index=False
)

print("\nSaved:")
print(OUT_METRICS)


# ============================================================
# PLOT 1 — IRRADIANCE RMSE
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(irr_results))

plt.plot(
    x,
    irr_results["Physics_RMSE_kW"],
    marker="o",
    label="Physics"
)

plt.plot(
    x,
    irr_results["PureML_RMSE_kW"],
    marker="o",
    label="Pure ML"
)

plt.plot(
    x,
    irr_results["Hybrid_RMSE_kW"],
    marker="o",
    label="Hybrid Residual"
)

plt.xticks(x, irr_results["regime"])

plt.xlabel("POA Irradiance (W/m²)")
plt.ylabel("RMSE (kW)")
plt.title("PVDAQ 2025 RMSE Across Irradiance Regimes")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "pvdaq_rmse_irradiance.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — TEMPERATURE RMSE
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(temp_results))

plt.plot(
    x,
    temp_results["Physics_RMSE_kW"],
    marker="o",
    label="Physics"
)

plt.plot(
    x,
    temp_results["PureML_RMSE_kW"],
    marker="o",
    label="Pure ML"
)

plt.plot(
    x,
    temp_results["Hybrid_RMSE_kW"],
    marker="o",
    label="Hybrid Residual"
)

plt.xticks(x, temp_results["regime"])

plt.xlabel("Cell Temperature (°C)")
plt.ylabel("RMSE (kW)")
plt.title("PVDAQ 2025 RMSE Across Temperature Regimes")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "pvdaq_rmse_temperature.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 — WIND RMSE
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(wind_results))

plt.plot(
    x,
    wind_results["Physics_RMSE_kW"],
    marker="o",
    label="Physics"
)

plt.plot(
    x,
    wind_results["PureML_RMSE_kW"],
    marker="o",
    label="Pure ML"
)

plt.plot(
    x,
    wind_results["Hybrid_RMSE_kW"],
    marker="o",
    label="Hybrid Residual"
)

plt.xticks(x, wind_results["regime"])

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("RMSE (kW)")
plt.title("PVDAQ 2025 RMSE Across Wind Regimes")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "pvdaq_rmse_wind.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4 — HYBRID IMPROVEMENT
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(irr_results))

plt.bar(
    x,
    irr_results["Hybrid_vs_Physics_improvement_pct"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(x, irr_results["regime"])

plt.xlabel("POA Irradiance (W/m²)")
plt.ylabel("Hybrid RMSE Improvement vs Physics (%)")
plt.title("Hybrid Improvement Across Irradiance Regimes")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "pvdaq_hybrid_improvement_irradiance.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 70)
print("PVDAQ ERROR-REGIME ANALYSIS COMPLETE")
print("=" * 70)