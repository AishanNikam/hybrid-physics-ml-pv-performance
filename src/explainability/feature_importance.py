import os
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


DATA_PATH = "data/processed/PV_simulation_clean.csv"
OUTPUT_CSV = "results/metrics/feature_importance.csv"
OUTPUT_FIG = "results/figures/feature_importance.png"


FEATURES = [
    "irradiance_wm2",
    "temperature_c",
    "shading_fraction"
]

TARGET = "p_max_global_w"


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Use the same IID training split
# ---------------------------------------------------------

train = df.sample(frac=0.8, random_state=42)

X_train = train[FEATURES]

y_train = (
    train[TARGET] -
    train["p_physics_w"]
)


# ---------------------------------------------------------
# 3. Train residual model
# ---------------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. Extract feature importance
# ---------------------------------------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": importance
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(importance_df.to_string(index=False))


# ---------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ---------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Feature")
plt.ylabel("XGBoost Importance")
plt.title("Feature Importance for Hybrid Residual Model")

plt.tight_layout()

plt.savefig(
    OUTPUT_FIG,
    dpi=300
)

plt.close()


print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_FIG)