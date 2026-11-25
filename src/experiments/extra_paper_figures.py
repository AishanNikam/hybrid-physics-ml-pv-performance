"""
Extra paper figures/tables:
  fig_feature_importance  - XGBoost gain importance of the hybrid residual model
  fig_mask_types          - physics error and hybrid error by structured shading pattern
  results/metrics/pvmismatch/mask_type_errors.csv

Run from repo root after run_pvmismatch.py and demo/train_demo_models.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures" / "paper"
SIM = ROOT / "results" / "metrics" / "pvmismatch"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE, GREY, ORANGE = "#2a78d6", "#7a7974", "#eb6834"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png"); plt.close(fig)
    print("saved", name)


def feature_importance():
    m = XGBRegressor(); m.load_model(ROOT / "demo" / "models" / "hybrid_residual.json")  # +string inputs
    gain = m.get_booster().get_score(importance_type="gain")
    s = pd.Series(gain).sort_values()
    s = (s / s.sum() * 100).tail(15)   # top 15 of all inputs (shares are of total gain)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    cols = [BLUE if k.startswith("str") or "string" in k else ("#9cc2ee" if k.startswith("S") else GREY) for k in s.index]
    ax.barh(s.index, s.values, color=cols, height=0.7)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("Hybrid residual model: top-15 inputs by gain\n(dark blue = per-string inputs, light = per-panel S1-S12)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_feature_importance")
    return s


def mask_types():
    p = pd.read_csv(SIM / "pvmismatch_predictions_seed42.csv")
    p2 = pd.read_csv(SIM.parent / "pvmismatch_string" / "pvmismatch_predictions_seed42.csv")
    p["pred_hybrid_str"] = p2["pred_hybrid"].values
    assert np.allclose(p.p_reference_w.values, p2.p_reference_w.values)
    p["pattern"] = (p.mask_type.str.replace(r"_sev\d+", "", regex=True)
                    .str.replace(r"_p\d+", "", regex=True).str.replace(r"2$", "", regex=True)
                    .str.replace("structured_", ""))
    names = {"random": "Random (0-4 panels)", "single": "Single panel", "two_same_str": "2 panels, same string",
             "two_diff_str": "2 panels, diff. strings", "four_one_str": "4 panels, one string",
             "four_all_str": "4 panels, spread", "uniform": "Uniform (all 12)", "gradient": "Gradient",
             "no_shading": "No shading"}
    rows = []
    for k, g in p.groupby("pattern"):
        y = g.p_reference_w
        rows.append({"pattern": names.get(k, k), "n": len(g),
                     "phys_bias_pct": 100 * ((g.pred_physics - y) / y).mean(),
                     "physics": np.sqrt(((g.pred_physics - y) ** 2).mean()),
                     "pure_ml": np.sqrt(((g.pred_pure_ml - y) ** 2).mean()),
                     "hybrid": np.sqrt(((g.pred_hybrid - y) ** 2).mean()),
                     "hybrid_str": np.sqrt(((g.pred_hybrid_str - y) ** 2).mean())})
    t = pd.DataFrame(rows).sort_values("physics", ascending=False)
    t.to_csv(SIM / "mask_type_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    y = np.arange(len(t))
    for off, col, c, lab in [(-0.3, "physics", GREY, "Physics"), (-0.1, "pure_ml", ORANGE, "Pure ML"),
                             (0.1, "hybrid", "#9cc2ee", "Hybrid (base inputs)"),
                             (0.3, "hybrid_str", BLUE, "Hybrid (+string inputs)")]:
        ax.barh(y + off, t[col], 0.19, color=c, label=lab)
    ax.set_yticks(y, [f"{a} ({n})" for a, n in zip(t.pattern, t.n)])
    ax.invert_yaxis(); ax.set_xlabel("RMSE (W)")
    ax.set_title("Error by shading pattern (test rows, seed 42)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.14)); ax.grid(axis="y", visible=False)
    save(fig, "fig_mask_types")
    return t


if __name__ == "__main__":
    print(feature_importance().round(1).to_string())
    print(mask_types().round(1).to_string(index=False))
