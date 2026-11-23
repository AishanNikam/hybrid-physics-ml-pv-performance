from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "PV_simulation_clean.csv"
OUTPUT_DIR = ROOT / "results" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "multiseed_iid_results.csv"


# ============================================================
# SETTINGS
# ============================================================

FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"

SEEDS = [42, 7, 21, 100, 123]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nrmse = rmse / y_true.mean() * 100

    return mae, rmse, r2, nrmse


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA)

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# RUN MULTIPLE SEEDS
# ============================================================

all_results = []

for seed in SEEDS:

    print()
    print("=" * 50)
    print(f"Running seed: {seed}")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed
    )

    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    physics_pred = df.loc[
        X_test.index,
        "p_physics_w"
    ]

    mae, rmse, r2, nrmse = calculate_metrics(
        y_test,
        physics_pred
    )

    all_results.append({
        "seed": seed,
        "model": "Physics",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse
    })

    # --------------------------------------------------------
    # Pure ML
    # --------------------------------------------------------

    print("Training Pure ML...")

    pure_ml = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        objective="reg:squarederror"
    )

    pure_ml.fit(X_train, y_train)

    pure_pred = pure_ml.predict(X_test)

    mae, rmse, r2, nrmse = calculate_metrics(
        y_test,
        pure_pred
    )

    all_results.append({
        "seed": seed,
        "model": "Pure ML",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse
    })

    # --------------------------------------------------------
    # Hybrid Residual
    # --------------------------------------------------------

    print("Training Hybrid Residual...")

    physics_train = df.loc[
        X_train.index,
        "p_physics_w"
    ]

    physics_test = df.loc[
        X_test.index,
        "p_physics_w"
    ]

    residual_train = (
        y_train - physics_train
    )

    hybrid = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        objective="reg:squarederror"
    )

    hybrid.fit(
        X_train,
        residual_train
    )

    residual_pred = hybrid.predict(X_test)

    hybrid_pred = (
        physics_test.values + residual_pred
    )

    mae, rmse, r2, nrmse = calculate_metrics(
        y_test,
        hybrid_pred
    )

    all_results.append({
        "seed": seed,
        "model": "Hybrid Residual",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE_percent": nrmse
    })


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        NRMSE_mean=("NRMSE_percent", "mean"),
        NRMSE_std=("NRMSE_percent", "std")
    )
    .reset_index()
)


print()
print("=" * 70)
print("MULTI-SEED RESULTS")
print("=" * 70)

print(
    summary.round(4).to_string(index=False)
)

print()
print("=" * 70)
print("RESULTS BY SEED")
print("=" * 70)

print(
    results.round(4).to_string(index=False)
)

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(f"Raw results: {OUTPUT}")