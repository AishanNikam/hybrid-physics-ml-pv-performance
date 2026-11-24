
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "results" / "metrics" / "pvdaq_error_regime_analysis.csv"
OUTPUT_DIR = ROOT / "results" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "regime_type",
    "regime",
    "n_samples",
    "Physics_RMSE_kW",
    "PureML_RMSE_kW",
    "Hybrid_RMSE_kW",
    "Hybrid_vs_Physics_improvement_pct",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", df.shape)
print("Regimes:", df["regime_type"].unique())


# ---------------------------------------------------------
# COMMON SETTINGS
# ---------------------------------------------------------

MODEL_COLUMNS = [
    "Physics_RMSE_kW",
    "PureML_RMSE_kW",
    "Hybrid_RMSE_kW",
]

MODEL_LABELS = [
    "Physics-only",
    "Pure ML",
    "Hybrid residual",
]

REGIME_NAMES = {
    "irradiance": "Irradiance",
    "temperature": "Cell temperature",
    "wind": "Wind speed",
    "hour": "Time of day",
}


# ---------------------------------------------------------
# FIGURES 1–4: RMSE COMPARISON BY REGIME
# ---------------------------------------------------------

for regime_type, title in REGIME_NAMES.items():

    subset = df[df["regime_type"] == regime_type].copy()

    if subset.empty:
        print(f"Skipping {regime_type}: no data found.")
        continue

    x = np.arange(len(subset))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (column, label) in enumerate(
        zip(MODEL_COLUMNS, MODEL_LABELS)
    ):

        values = subset[column].to_numpy()

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=label,
        )

        # Value labels
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Include sample counts in category labels
    tick_labels = [
        f"{regime}\n(n={n})"
        for regime, n in zip(
            subset["regime"],
            subset["n_samples"],
        )
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel(title)
    ax.set_ylabel("RMSE (kW)")
    ax.set_title(
        f"PVDAQ External Validation: Model RMSE by {title} Regime"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    max_value = subset[MODEL_COLUMNS].to_numpy().max()
    ax.set_ylim(0, max_value * 1.20)

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"pvdaq_rmse_{regime_type}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_file)


# ---------------------------------------------------------
# FIGURE 5: HYBRID IMPROVEMENT OVER PHYSICS BASELINE
# ---------------------------------------------------------

plot_df = df.copy()

plot_df["category_label"] = (
    plot_df["regime_type"].map(REGIME_NAMES)
    + " | "
    + plot_df["regime"].astype(str)
    + " (n="
    + plot_df["n_samples"].astype(str)
    + ")"
)

# Preserve logical grouping and ordering
regime_order = ["irradiance", "temperature", "wind", "hour"]

plot_df["regime_type"] = pd.Categorical(
    plot_df["regime_type"],
    categories=regime_order,
    ordered=True,
)

plot_df = plot_df.sort_values(
    ["regime_type"],
    kind="stable",
).iloc[::-1]

values = plot_df[
    "Hybrid_vs_Physics_improvement_pct"
].to_numpy()

labels = plot_df["category_label"].to_numpy()

# Distinguish improvement from degradation
bar_colors = [
    "tab:blue" if value >= 0 else "tab:orange"
    for value in values
]

fig, ax = plt.subplots(figsize=(12, 10))

bars = ax.barh(
    labels,
    values,
    color=bar_colors,
)

ax.axvline(
    0,
    color="black",
    linewidth=1,
)

for bar, value in zip(bars, values):

    offset = 0.4 if value >= 0 else -0.4

    ax.annotate(
        f"{value:+.1f}%",
        xy=(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
        ),
        ha="left" if value >= 0 else "right",
        va="center",
        fontsize=8,
    )

ax.set_xlabel("RMSE reduction relative to physics-only (%)")
ax.set_title(
    "PVDAQ External Validation: Hybrid Model Improvement by Regime"
)

ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

plt.tight_layout()

output_file = OUTPUT_DIR / "pvdaq_hybrid_improvement_by_regime.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output_file)


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PVDAQ VISUALISATION COMPLETE")
print("=" * 65)

print("\nFigures saved in:")
print(OUTPUT_DIR)

for file in sorted(OUTPUT_DIR.glob("pvdaq_*.png")):
    print(file.name)
