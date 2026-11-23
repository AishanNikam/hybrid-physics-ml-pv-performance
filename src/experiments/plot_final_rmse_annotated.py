import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_PATH = "results/metrics/final_model_comparison.csv"
OUTPUT_PATH = "results/figures/final_rmse_comparison_annotated.png"

df = pd.read_csv(INPUT_PATH)

x = range(len(df))
width = 0.25

plt.figure(figsize=(12, 6))

bars1 = plt.bar(
    [i - width for i in x],
    df["Physics_RMSE"],
    width=width,
    label="Physics-only"
)

bars2 = plt.bar(
    x,
    df["PureML_RMSE"],
    width=width,
    label="Pure ML"
)

bars3 = plt.bar(
    [i + width for i in x],
    df["Hybrid_RMSE"],
    width=width,
    label="Hybrid Residual"
)

labels = [
    "IID",
    "Temperature\nOOD",
    "Irradiance\nOOD",
    "Shading\nOOD",
    "Combined\nOOD"
]

plt.xticks(list(x), labels)

plt.ylabel("RMSE")
plt.xlabel("Evaluation Regime")
plt.title("PV Power Prediction: Physics vs Pure ML vs Hybrid")

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

# Add numerical values
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.6,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.tight_layout()

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print(OUTPUT_PATH)

plt.show()