import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Find project root
# --------------------------------------------------

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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Select input features
# --------------------------------------------------

FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"


X = df[FEATURES]
y = df[TARGET]


# --------------------------------------------------
# 4. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 5. Physics-only prediction
# --------------------------------------------------

physics_prediction = df.loc[X_test.index, "p_physics_w"].values


# --------------------------------------------------
# 6. Pure ML model
# --------------------------------------------------

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

pure_ml.fit(X_train, y_train)

pure_ml_prediction = pure_ml.predict(X_test)


# --------------------------------------------------
# 7. Hybrid residual model
# --------------------------------------------------

print("Training Hybrid Residual model...")

physics_train = df.loc[X_train.index, "p_physics_w"].values

residual_train = (
    y_train.values - physics_train
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


# Predict residual on test data
residual_prediction = hybrid_model.predict(X_test)

# Hybrid = Physics + ML correction
hybrid_prediction = (
    physics_prediction + residual_prediction
)


# --------------------------------------------------
# 8. Evaluation function
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

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


# --------------------------------------------------
# 9. Calculate results
# --------------------------------------------------

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


# --------------------------------------------------
# 10. Create comparison table
# --------------------------------------------------

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


# --------------------------------------------------
# 11. Print results
# --------------------------------------------------

print("\n========================================")
print("IID MODEL COMPARISON")
print("========================================")

print(
    results.to_string(index=False)
)


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

output_path = (
    RESULTS_DIR / "iid_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")

print(output_path)