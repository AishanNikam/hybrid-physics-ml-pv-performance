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
# 3. DEFINE FEATURES
# ==================================================

FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"


# ==================================================
# 4. DEFINE TEMPERATURE OOD
# ==================================================

# Training temperatures:
# 20°C to 40°C

TRAIN_MAX_TEMP = 40

# OOD temperatures:
# 41°C to 50°C

train_df = df[
    df["temperature_c"] <= TRAIN_MAX_TEMP
].copy()

test_df = df[
    df["temperature_c"] > TRAIN_MAX_TEMP
].copy()


print("\n========================================")
print("TEMPERATURE OOD SPLIT")
print("========================================")

print(
    f"Training samples : {len(train_df)}"
)

print(
    f"OOD test samples : {len(test_df)}"
)

print(
    f"Training temperatures: "
    f"{train_df['temperature_c'].min()} - "
    f"{train_df['temperature_c'].max()} °C"
)

print(
    f"OOD temperatures: "
    f"{test_df['temperature_c'].min()} - "
    f"{test_df['temperature_c'].max()} °C"
)


# ==================================================
# 5. PREPARE X AND y
# ==================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ==================================================
# 6. PHYSICS-ONLY MODEL
# ==================================================

physics_prediction = (
    test_df["p_physics_w"].values
)


# ==================================================
# 7. PURE ML MODEL
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
# 8. HYBRID RESIDUAL MODEL
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


# Hybrid prediction
hybrid_prediction = (
    physics_prediction
    + residual_prediction
)


# ==================================================
# 9. METRICS FUNCTION
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
# 10. CALCULATE METRICS
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
# 11. COMPARISON TABLE
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
# 12. PRINT RESULTS
# ==================================================

print("\n========================================")
print("TEMPERATURE OOD RESULTS")
print("========================================")

print(
    results.to_string(index=False)
)


# ==================================================
# 13. SAVE RESULTS
# ==================================================

output_path = (
    RESULTS_DIR
    / "temperature_ood_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")

print(output_path)