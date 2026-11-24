from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import pvlib
from pvlib.pvsystem import retrieve_sam, calcparams_cec, singlediode
from pvlib.temperature import sapm_cell

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "PVDAQ_2107_clean.csv"

METRICS_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "poa_irradiance_wm2",
    "ambient_temperature_c",
    "t_cell",
    "wind_speed_ms",
]

TARGET = "dc_power_kw"

TRAIN_YEAR = 2024
TEST_YEAR = 2025

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("PVDAQ CHRONOLOGICAL HOLDOUT EVALUATION")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

print(f"\nLoaded dataset: {df.shape}")


# Keep only rows with measured DC power
df = df[df[TARGET].notna()].copy()

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Keep rows where required physical measurements exist
# Keep rows where required measured variables exist
df = df[
    df["poa_irradiance_wm2"].notna()
    & df["ambient_temperature_c"].notna()
    & df[TARGET].notna()
].copy()


# ============================================================
# CALCULATE CELL TEMPERATURE
# ============================================================

print("\nCalculating cell temperature using pvlib SAPM...")

# PV module temperature parameters.
# Open-rack glass-glass configuration.
# These are the standard SAPM temperature-model coefficients
# used for the project physics workflow.
temp_params = {
    "a": -3.47,
    "b": -0.0594,
    "deltaT": 3,
}

# Use training-year wind statistics for missing-value handling
train_wind_median = df.loc[
    df["timestamp"].dt.year == TRAIN_YEAR,
    "wind_speed_ms"
].median()

print(
    f"Training-year wind median: "
    f"{train_wind_median:.4f} m/s"
)

# Calculate cell temperature without using test-period
# statistics for imputing missing wind measurements
df["t_cell"] = sapm_cell(
    poa_global=df["poa_irradiance_wm2"].to_numpy(dtype=float),
    temp_air=df["ambient_temperature_c"].to_numpy(dtype=float),
    wind_speed=df["wind_speed_ms"].fillna(
        train_wind_median
    ).to_numpy(dtype=float),
    a=temp_params["a"],
    b=temp_params["b"],
    deltaT=temp_params["deltaT"],
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train = df[df["timestamp"].dt.year == TRAIN_YEAR].copy()
test = df[df["timestamp"].dt.year == TEST_YEAR].copy()

print("\nChronological split:")
print(f"Train: {len(train)} rows ({TRAIN_YEAR})")
print(f"Test : {len(test)} rows ({TEST_YEAR})")

print(
    f"\nTrain dates: {train['timestamp'].min()} -> "
    f"{train['timestamp'].max()}"
)

print(
    f"Test dates : {test['timestamp'].min()} -> "
    f"{test['timestamp'].max()}"
)


# ============================================================
# HANDLE MISSING WIND
# ============================================================

wind_median = train["wind_speed_ms"].median()

train["wind_speed_ms"] = train["wind_speed_ms"].fillna(wind_median)
test["wind_speed_ms"] = test["wind_speed_ms"].fillna(wind_median)

print(
    f"\nWind median used for missing values: "
    f"{wind_median:.4f} m/s"
)

# ============================================================
# PHYSICS BASELINE
# ============================================================

print("\n" + "-" * 70)
print("Building physics baseline")
print("-" * 70)

# ------------------------------------------------------------
# Canadian Solar CS6P-260P
# CEC module database
# ------------------------------------------------------------

cec_modules = retrieve_sam("CECMod")

module_name = None

module_name = "Canadian_Solar_Inc__CS6P_260P"

if module_name not in cec_modules.columns:
    raise RuntimeError(
        f"{module_name} was not found in the installed pvlib CEC database."
    )

if module_name is None:
    raise RuntimeError(
        "Canadian Solar CS6P-260P was not found in the installed "
        "pvlib CEC module database."
    )

module = cec_modules[module_name]

print(f"Module selected: {module_name}")


# ------------------------------------------------------------
# Module STC rating
# ------------------------------------------------------------

p_stc_module_w = float(module["STC"])

print(f"Module STC power: {p_stc_module_w:.2f} W")


# ------------------------------------------------------------
# Estimate module count from 2024 measured data
#
# IMPORTANT:
# This is used ONLY to establish system scale.
# It is NOT fitted using 2025 test data.
#
# We use the maximum observed 2024 DC power.
# ------------------------------------------------------------

max_train_power_kw = train[TARGET].max()

n_modules = int(
    np.ceil(
        (max_train_power_kw * 1000) / p_stc_module_w
    )
)

system_stc_kw = (
    n_modules * p_stc_module_w / 1000.0
)

print(f"Maximum 2024 measured DC power: {max_train_power_kw:.3f} kW")
print(f"Estimated module count: {n_modules}")
print(f"Estimated system STC capacity: {system_stc_kw:.3f} kW")


# ============================================================
# FUNCTION: PHYSICS PREDICTION
# ============================================================

def physics_prediction(data):
    """
    Generate system-level PV power using pvlib CEC single-diode model.

    Inputs:
        POA irradiance
        cell temperature

    Output:
        predicted system DC power in kW

    No measured DC power from the prediction row is used.
    """

    effective_irradiance = np.maximum(
        data["poa_irradiance_wm2"].to_numpy(dtype=float),
        0.0
    )

    temp_cell = data["t_cell"].to_numpy(dtype=float)

    # CEC single-diode parameter calculation
    IL, I0, Rs, Rsh, nNsVth = calcparams_cec(
        effective_irradiance,
        temp_cell,
        alpha_sc=module["alpha_sc"],
        a_ref=module["a_ref"],
        I_L_ref=module["I_L_ref"],
        I_o_ref=module["I_o_ref"],
        R_sh_ref=module["R_sh_ref"],
        R_s=module["R_s"],
        Adjust=module["Adjust"],
    )

    # Solve single-diode equation
    result = singlediode(
        IL,
        I0,
        Rs,
        Rsh,
        nNsVth,
    )

    p_mp_module_w = np.maximum(
        np.asarray(result["p_mp"], dtype=float),
        0.0
    )

    # Scale module power to system
    p_system_kw = (
        p_mp_module_w * n_modules / 1000.0
    )

    return p_system_kw


# ============================================================
# GENERATE PHYSICS PREDICTIONS
# ============================================================

train["physics_kw"] = physics_prediction(train)
test["physics_kw"] = physics_prediction(test)

print("\nPhysics predictions generated.")


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()

y_train = train[TARGET].to_numpy()
y_test = test[TARGET].to_numpy()

physics_train = train["physics_kw"].to_numpy()
physics_test = test["physics_kw"].to_numpy()


# ============================================================
# PURE ML
# ============================================================

print("\n" + "-" * 70)
print("Training Pure ML model")
print("-" * 70)

pure_ml = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
)

pure_ml.fit(X_train, y_train)

pure_ml_pred = pure_ml.predict(X_test)


# ============================================================
# HYBRID RESIDUAL ML
# ============================================================

print("\n" + "-" * 70)
print("Training Hybrid Residual model")
print("-" * 70)

residual_train = y_train - physics_train

hybrid_ml = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    objective="reg:squarederror",
)

hybrid_ml.fit(X_train, residual_train)

residual_pred = hybrid_ml.predict(X_test)

hybrid_pred = physics_test + residual_pred


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, pred):

    mae = mean_absolute_error(y_true, pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, pred)
    )

    r2 = r2_score(y_true, pred)

    nrmse = (
        rmse / np.mean(y_true) * 100
    )

    bias = np.mean(pred - y_true)

    return {
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse,
        "Bias_kW": bias,
    }


results = []

for name, pred in [
    ("Physics", physics_test),
    ("Pure ML", pure_ml_pred),
    ("Hybrid Residual", hybrid_pred),
]:

    metrics = calculate_metrics(
        y_test,
        pred
    )

    metrics["model"] = name
    metrics["train_year"] = TRAIN_YEAR
    metrics["test_year"] = TEST_YEAR
    metrics["n_test"] = len(y_test)

    results.append(metrics)


results_df = pd.DataFrame(results)

# Arrange columns
results_df = results_df[
    [
        "model",
        "MAE_kW",
        "RMSE_kW",
        "R2",
        "NRMSE_percent",
        "Bias_kW",
        "train_year",
        "test_year",
        "n_test",
    ]
]


# ============================================================
# HYBRID IMPROVEMENT
# ============================================================

physics_rmse = results_df.loc[
    results_df["model"] == "Physics",
    "RMSE_kW"
].iloc[0]

hybrid_rmse = results_df.loc[
    results_df["model"] == "Hybrid Residual",
    "RMSE_kW"
].iloc[0]

pure_ml_rmse = results_df.loc[
    results_df["model"] == "Pure ML",
    "RMSE_kW"
].iloc[0]

hybrid_vs_physics = (
    (physics_rmse - hybrid_rmse)
    / physics_rmse
    * 100
)

hybrid_vs_pure = (
    (pure_ml_rmse - hybrid_rmse)
    / pure_ml_rmse
    * 100
)


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_path = (
    METRICS_DIR /
    "pvdaq_external_validation_results.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test[
    [
        "timestamp",
        "poa_irradiance_wm2",
        "ambient_temperature_c",
        "t_cell",
        "wind_speed_ms",
        TARGET,
    ]
].copy()

predictions.rename(
    columns={
        TARGET: "Measured_DC_Power_kW"
    },
    inplace=True
)

predictions["Physics_Prediction_kW"] = physics_test
predictions["PureML_Prediction_kW"] = pure_ml_pred
predictions["Hybrid_Prediction_kW"] = hybrid_pred

predictions["Physics_Error_kW"] = (
    predictions["Physics_Prediction_kW"]
    - predictions["Measured_DC_Power_kW"]
)

predictions["PureML_Error_kW"] = (
    predictions["PureML_Prediction_kW"]
    - predictions["Measured_DC_Power_kW"]
)

predictions["Hybrid_Error_kW"] = (
    predictions["Hybrid_Prediction_kW"]
    - predictions["Measured_DC_Power_kW"]
)

pred_path = (
    PRED_DIR /
    "pvdaq_2025_predictions.csv"
)

predictions.to_csv(
    pred_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PVDAQ CHRONOLOGICAL HOLDOUT RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "MAE_kW",
            "RMSE_kW",
            "R2",
            "NRMSE_percent",
            "Bias_kW",
        ]
    ].round(4).to_string(index=False)
)

print("\nHybrid vs Physics RMSE improvement:")
print(f"  {hybrid_vs_physics:.2f}%")

print("\nHybrid vs Pure ML RMSE improvement:")
print(f"  {hybrid_vs_pure:.2f}%")

print("\nSaved:")
print(f"  {metrics_path}")
print(f"  {pred_path}")

print("\nPVDAQ validation complete.")
