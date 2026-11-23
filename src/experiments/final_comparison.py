import os
import pandas as pd

METRICS_DIR = "results/metrics"
OUTPUT_PATH = "results/metrics/final_model_comparison.csv"

files = {
    "IID": "iid_results.csv",
    "Temperature OOD": "temperature_ood_results.csv",
    "Irradiance OOD": "irradiance_ood_results.csv",
    "Shading OOD": "shading_ood_results.csv",
    "Combined OOD": "combined_ood_results.csv"
}

all_results = []

for experiment, filename in files.items():

    path = os.path.join(METRICS_DIR, filename)

    print(f"\nReading: {filename}")

    df = pd.read_csv(path)

    # -----------------------------------------
    # Normalize column names
    # -----------------------------------------

    rename_map = {
        "Model": "model",
        "MAE_W": "MAE",
        "RMSE_W": "RMSE"
    }

    df = df.rename(columns=rename_map)

    required = ["model", "MAE", "RMSE", "R2", "NRMSE_percent"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required].copy()

    df["experiment"] = experiment

    all_results.append(df)

# -----------------------------------------
# Combine all experiments
# -----------------------------------------

results = pd.concat(all_results, ignore_index=True)

# -----------------------------------------
# RMSE table
# -----------------------------------------

rmse_table = results.pivot_table(
    index="experiment",
    columns="model",
    values="RMSE",
    aggfunc="first"
).reset_index()

rmse_table.columns.name = None

rmse_table = rmse_table.rename(columns={
    "Physics-only": "Physics_RMSE",
    "Pure ML": "PureML_RMSE",
    "Hybrid Residual": "Hybrid_RMSE"
})

# -----------------------------------------
# R2 table
# -----------------------------------------

r2_table = results.pivot_table(
    index="experiment",
    columns="model",
    values="R2",
    aggfunc="first"
).reset_index()

r2_table.columns.name = None

r2_table = r2_table.rename(columns={
    "Physics-only": "Physics_R2",
    "Pure ML": "PureML_R2",
    "Hybrid Residual": "Hybrid_R2"
})

# -----------------------------------------
# Merge RMSE + R2
# -----------------------------------------

final = rmse_table.merge(
    r2_table[
        [
            "experiment",
            "Physics_R2",
            "PureML_R2",
            "Hybrid_R2"
        ]
    ],
    on="experiment"
)

# -----------------------------------------
# Calculate improvement
# -----------------------------------------

final["Hybrid_vs_Physics_%"] = (
    (final["Physics_RMSE"] - final["Hybrid_RMSE"])
    / final["Physics_RMSE"]
    * 100
)

final["Hybrid_vs_PureML_%"] = (
    (final["PureML_RMSE"] - final["Hybrid_RMSE"])
    / final["PureML_RMSE"]
    * 100
)

# -----------------------------------------
# Arrange columns
# -----------------------------------------

final = final[
    [
        "experiment",

        "Physics_RMSE",
        "PureML_RMSE",
        "Hybrid_RMSE",

        "Physics_R2",
        "PureML_R2",
        "Hybrid_R2",

        "Hybrid_vs_Physics_%",
        "Hybrid_vs_PureML_%"
    ]
]

# -----------------------------------------
# Experiment order
# -----------------------------------------

experiment_order = [
    "IID",
    "Temperature OOD",
    "Irradiance OOD",
    "Shading OOD",
    "Combined OOD"
]

final["experiment"] = pd.Categorical(
    final["experiment"],
    categories=experiment_order,
    ordered=True
)

final = final.sort_values("experiment")

# -----------------------------------------
# Print final comparison
# -----------------------------------------

print("\n")
print("========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    final.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# -----------------------------------------
# Save
# -----------------------------------------

os.makedirs(METRICS_DIR, exist_ok=True)

final.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved to:")
print(OUTPUT_PATH)