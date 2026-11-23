from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. Project paths
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
    / "predictions"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Features and target
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
# 4. IID train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 5. Physics prediction
# --------------------------------------------------

physics_prediction = (
    df.loc[X_test.index, "p_physics_w"].values
)


# --------------------------------------------------
# 6. Pure ML
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

pure_ml.fit(
    X_train,
    y_train
)

pure_ml_prediction = pure_ml.predict(X_test)


# --------------------------------------------------
# 7. Hybrid residual model
# --------------------------------------------------

print("Training Hybrid Residual model...")

physics_train = (
    df.loc[X_train.index, "p_physics_w"].values
)

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

residual_prediction = hybrid_model.predict(X_test)

hybrid_prediction = (
    physics_prediction + residual_prediction
)


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

predictions = df.loc[
    X_test.index,
    [
        "irradiance_wm2",
        "temperature_c",
        "shading_fraction",
        "p_physics_w",
        "p_max_global_w"
    ]
].copy()

predictions = predictions.rename(
    columns={
        "p_physics_w": "Physics_Prediction_W",
        "p_max_global_w": "Actual_Power_W"
    }
)

predictions["PureML_Prediction_W"] = (
    pure_ml_prediction
)

predictions["Hybrid_Prediction_W"] = (
    hybrid_prediction
)

predictions["Physics_Error_W"] = (
    predictions["Actual_Power_W"]
    - predictions["Physics_Prediction_W"]
)

predictions["PureML_Error_W"] = (
    predictions["Actual_Power_W"]
    - predictions["PureML_Prediction_W"]
)

predictions["Hybrid_Error_W"] = (
    predictions["Actual_Power_W"]
    - predictions["Hybrid_Prediction_W"]
)

# Keep original test-set indices
predictions["original_index"] = X_test.index

predictions = predictions[
    [
        "original_index",
        "irradiance_wm2",
        "temperature_c",
        "shading_fraction",
        "Actual_Power_W",
        "Physics_Prediction_W",
        "PureML_Prediction_W",
        "Hybrid_Prediction_W",
        "Physics_Error_W",
        "PureML_Error_W",
        "Hybrid_Error_W"
    ]
]


# --------------------------------------------------
# 9. Save
# --------------------------------------------------

output_path = (
    RESULTS_DIR / "iid_predictions.csv"
)

predictions.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("PREDICTIONS SAVED")
print("========================================")

print(
    predictions.head(10).to_string(index=False)
)

print("\nNumber of test predictions:")
print(len(predictions))

print("\nSaved to:")
print(output_path)