import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "PV_simulation_clean.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ==================================================
# 3. FEATURES
# ==================================================

FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"


# ==================================================
# 4. IRRADIANCE OOD SPLIT
# ==================================================

# IMPORTANT:
# We use ONLY unshaded samples.
#
# Training:
#   Irradiance = 500–900 W/m²
#   Shading = 0%
#
# OOD:
#   Irradiance = 950–1000 W/m²
#   Shading = 0%
#
# This isolates irradiance OOD from shading OOD.


train_df = df[
    (df["shading_pct"] == 0)
    &
    (df["irradiance_wm2"] <= 900)
].copy()


test_df = df[
    (df["shading_pct"] == 0)
    &
    (df["irradiance_wm2"] >= 950)
].copy()


# ==================================================
# 5. PRINT SPLIT INFORMATION
# ==================================================

print("\n========================================")
print("IRRADIANCE OOD SPLIT")
print("========================================")

print(
    f"Training samples : {len(train_df)}"
)

print(
    f"OOD test samples : {len(test_df)}"
)

print(
    f"Training irradiance: "
    f"{train_df['irradiance_wm2'].min()} - "
    f"{train_df['irradiance_wm2'].max()} W/m²"
)

print(
    f"OOD irradiance: "
    f"{test_df['irradiance_wm2'].min()} - "
    f"{test_df['irradiance_wm2'].max()} W/m²"
)

print(
    f"Training shading: "
    f"{train_df['shading_pct'].unique()}"
)

print(
    f"Test shading: "
    f"{test_df['shading_pct'].unique()}"
)


# ==================================================
# 6. PREPARE TRAIN / TEST
# ==================================================

X_train = train_df[FEATURES]

y_train = train_df[TARGET]

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


# ==================================================
# 7. PHYSICS-ONLY
# ==================================================

physics_prediction = (
    test_df["p_physics_w"].values
)


# ==================================================
# 8. PURE ML
# ==================================================

print("\nTraining Pure ML model...")

pure_ml = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

pure_ml.fit(
    X_train,
    y_train
)

pure_ml_prediction = (
    pure_ml.predict(X_test)
)


# ==================================================
# 9. HYBRID RESIDUAL MODEL
# ==================================================

print("Training Hybrid Residual model...")

physics_train = (
    train_df["p_physics_w"].values
)

residual_train = (
    y_train.values
    - physics_train
)


hybrid_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

hybrid_model.fit(
    X_train,
    residual_train
)


# Predict physics error

residual_prediction = (
    hybrid_model.predict(X_test)
)


# Physics + learned correction

hybrid_prediction = (
    physics_prediction
    + residual_prediction
)


# ==================================================
# 10. METRICS
# ==================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = (
        mean_squared_error(
            y_true,
            y_pred
        ) ** 0.5
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    nrmse = (
        rmse / y_true.mean()
    ) * 100

    return {
        "MAE_W": mae,
        "RMSE_W": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse
    }


# ==================================================
# 11. CALCULATE RESULTS
# ==================================================

physics_metrics = calculate_metrics(
    y_test,
    physics_prediction
)

pure_ml_metrics = calculate_metrics(
    y_test,
    pure_ml_prediction
)

hybrid_metrics = calculate_metrics(
    y_test,
    hybrid_prediction
)


# ==================================================
# 12. RESULTS TABLE
# ==================================================

results = pd.DataFrame([
    {
        "Model": "Physics-only",
        **physics_metrics
    },

    {
        "Model": "Pure ML",
        **pure_ml_metrics
    },

    {
        "Model": "Hybrid Residual",
        **hybrid_metrics
    }
])


# ==================================================
# 13. PRINT RESULTS
# ==================================================

print("\n========================================")
print("IRRADIANCE OOD RESULTS")
print("========================================")

print(
    results.to_string(index=False)
)


# ==================================================
# 14. SAVE RESULTS
# ==================================================

output_path = (
    RESULTS_DIR
    / "irradiance_ood_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")

print(output_path)