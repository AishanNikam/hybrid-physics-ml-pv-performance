import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = "data/processed/PV_simulation_clean.csv"
OUTPUT_PATH = "results/metrics/ablation_results.csv"

TARGET = "p_max_global_w"

ALL_FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nrmse = (rmse / np.mean(y_true)) * 100

    return mae, rmse, r2, nrmse


def run_hybrid(train, test, features):

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    physics_train = train["p_physics_w"]
    physics_test = test["p_physics_w"]

    # Learn physics residual
    residual_train = y_train - physics_train

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, residual_train)

    residual_pred = model.predict(X_test)

    hybrid_pred = physics_test + residual_pred

    return calculate_metrics(y_test, hybrid_pred)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# Define experiments
# ---------------------------------------------------------

feature_sets = {
    "G + T + Shading": [
        "irradiance_wm2",
        "temperature_c",
        "shading_fraction"
    ],

    "G + T": [
        "irradiance_wm2",
        "temperature_c"
    ],

    "G + Shading": [
        "irradiance_wm2",
        "shading_fraction"
    ],

    "T + Shading": [
        "temperature_c",
        "shading_fraction"
    ],

    "G only": [
        "irradiance_wm2"
    ],

    "T only": [
        "temperature_c"
    ],

    "Shading only": [
        "shading_fraction"
    ]
}


# ---------------------------------------------------------
# Define train/test splits
# ---------------------------------------------------------

splits = {

    "IID": (
        df.sample(frac=0.8, random_state=42),
        df.drop(df.sample(frac=0.8, random_state=42).index)
    ),

    "Temperature OOD": (
        df[df["temperature_c"] <= 40],
        df[df["temperature_c"] > 40]
    ),

    "Irradiance OOD": (
        df[
            (df["shading_pct"] == 0) &
            (df["irradiance_wm2"] <= 900)
        ],

        df[
            (df["shading_pct"] == 0) &
            (df["irradiance_wm2"] >= 950)
        ]
    ),

    "Shading OOD": (
        df[
            (df["irradiance_wm2"] == 1000) &
            (df["shading_pct"] <= 30)
        ],

        df[
            (df["irradiance_wm2"] == 1000) &
            (df["shading_pct"] >= 40)
        ]
    ),

    "Combined OOD": (
        df[
            (df["irradiance_wm2"] == 1000) &
            (df["shading_pct"] <= 30)
        ],

        df[
            (df["irradiance_wm2"] == 1000) &
            (df["shading_pct"] >= 40) &
            (df["temperature_c"] >= 41)
        ]
    )
}


# ---------------------------------------------------------
# Run ablation
# ---------------------------------------------------------

results = []

for split_name, (train, test) in splits.items():

    print("\n========================================")
    print(split_name)
    print("Train:", len(train))
    print("Test :", len(test))
    print("========================================")

    for feature_name, features in feature_sets.items():

        mae, rmse, r2, nrmse = run_hybrid(
            train,
            test,
            features
        )

        results.append({
            "experiment": split_name,
            "features": feature_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "NRMSE_percent": nrmse
        })

        print(
            f"{feature_name:20s} "
            f"RMSE={rmse:8.3f} "
            f"R2={r2:8.3f}"
        )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("results/metrics", exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========================================")
print("ABLATION COMPLETE")
print("========================================")

print(results_df.to_string(index=False))

print("\nSaved to:")
print(OUTPUT_PATH)