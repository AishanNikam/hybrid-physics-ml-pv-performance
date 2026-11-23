import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = "data/processed/PV_simulation_clean.csv"
OUTPUT_PATH = "results/metrics/combined_ood_results.csv"

FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"


def calculate_metrics(y_true, y_pred, name):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    nrmse = (rmse / np.mean(y_true)) * 100

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse
    }


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Define combined OOD split
# ---------------------------------------------------------
#
# Training:
#   Irradiance = 1000 W/m²
#   Shading = 0–30%
#   Temperature = 20–50°C
#
# Test:
#   Irradiance = 1000 W/m²
#   Shading = 40–50%
#   Temperature = 41–50°C
#
# The test region represents a joint
# severe-shading + high-temperature regime.
# ---------------------------------------------------------

train = df[
    (df["irradiance_wm2"] == 1000) &
    (df["shading_pct"] <= 30)
].copy()

test = df[
    (df["irradiance_wm2"] == 1000) &
    (df["shading_pct"] >= 40) &
    (df["temperature_c"] >= 41)
].copy()


print("\nTraining samples:", len(train))
print("Test samples:", len(test))

print(
    "\nTraining shading:",
    sorted(train["shading_pct"].unique())
)

print(
    "Test shading:",
    sorted(test["shading_pct"].unique())
)

print(
    "Training temperature:",
    train["temperature_c"].min(),
    "to",
    train["temperature_c"].max()
)

print(
    "Test temperature:",
    test["temperature_c"].min(),
    "to",
    test["temperature_c"].max()
)


# ---------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

physics_train = train["p_physics_w"]
physics_test = test["p_physics_w"]


# ---------------------------------------------------------
# 4. Pure ML model
# ---------------------------------------------------------

pure_ml = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pure_ml.fit(X_train, y_train)

pure_ml_pred = pure_ml.predict(X_test)


# ---------------------------------------------------------
# 5. Hybrid residual model
# ---------------------------------------------------------

residual_train = y_train - physics_train

hybrid = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

hybrid.fit(X_train, residual_train)

residual_pred = hybrid.predict(X_test)

hybrid_pred = physics_test + residual_pred


# ---------------------------------------------------------
# 6. Calculate metrics
# ---------------------------------------------------------

results = []

results.append(
    calculate_metrics(
        y_test,
        physics_test,
        "Physics-only"
    )
)

results.append(
    calculate_metrics(
        y_test,
        pure_ml_pred,
        "Pure ML"
    )
)

results.append(
    calculate_metrics(
        y_test,
        hybrid_pred,
        "Hybrid Residual"
    )
)


results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

print("\n========================================")
print("COMBINED OOD RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# ---------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------

os.makedirs("results/metrics", exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:", OUTPUT_PATH)