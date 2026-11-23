import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_PATH = "results/metrics/final_model_comparison.csv"
OUTPUT_PATH = "results/figures/final_rmse_comparison.png"

# -----------------------------------------
# Load results
# -----------------------------------------

df = pd.read_csv(INPUT_PATH)

# -----------------------------------------
# Plot
# -----------------------------------------

x = range(len(df))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width for i in x],
    df["Physics_RMSE"],
    width=width,
    label="Physics-only"
)

plt.bar(
    x,
    df["PureML_RMSE"],
    width=width,
    label="Pure ML"
)

plt.bar(
    [i + width for i in x],
    df["Hybrid_RMSE"],
    width=width,
    label="Hybrid Residual"
)

plt.xticks(
    list(x),
    [
        "IID",
        "Temperature\nOOD",
        "Irradiance\nOOD",
        "Shading\nOOD",
        "Combined\nOOD"
    ]
)

plt.ylabel("RMSE")
plt.xlabel("Evaluation Regime")
plt.title("PV Power Prediction: Physics vs Pure ML vs Hybrid")

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

# -----------------------------------------
# Save
# -----------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved figure:")
print(OUTPUT_PATH)

plt.show()