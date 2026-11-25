"""
Regime analysis (2025 locked test) + IEEE-ready figures for both datasets.

Inputs : results/metrics/pvmismatch/*, results/metrics/pvdaq_v2/*
Outputs: results/metrics/pvdaq_v2/test2025_regimes.csv
         results/figures/paper/fig*.pdf / .png   (3.5 in = IEEE single column)

Run from repo root (after run_pvmismatch.py and run_pvdaq_v2.py):
    python src/experiments/make_paper_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "results" / "metrics" / "pvmismatch_string"   # main model: with per-string inputs
FLD = ROOT / "results" / "metrics" / "pvdaq_v2"
FIG = ROOT / "results" / "figures" / "paper"
FIG.mkdir(parents=True, exist_ok=True)

# Model -> (label, colour, marker, linestyle). Colours: validated categorical
# slots 1-3 + neutral grey for the physics baseline; markers/linestyles make
# every series distinguishable in greyscale print.
STYLE = {
    "physics":    ("Physics (pvlib SDM)", "#7a7974", "s", ":"),
    "linear_res": ("Physics + linear corr.", "#1baf7a", "^", "-."),
    "pure_ml":    ("Pure ML (XGBoost)", "#eb6834", "o", "--"),
    "hybrid":     ("Hybrid (physics + XGBoost)", "#2a78d6", "D", "-"),
}
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
W1, W2 = 3.5, 7.16   # IEEE single / double column width (in)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


# ======================================================== PVDAQ regimes
def pvdaq_regimes():
    d = pd.read_csv(FLD / "test2025_predictions.csv", parse_dates=["timestamp"])
    y = d.dc_power_kw.values
    cols = {"physics": "pred_physics", "linear_res": "pred_linear_res [base+time]",
            "pure_ml": "pred_pure_ml [base+time]", "hybrid": "pred_hybrid [base+time]",
            "hybrid_base": "pred_hybrid [base]"}
    d["hour"] = d.timestamp.dt.hour
    bins = {
        "irradiance": pd.cut(d.poa_irradiance_wm2, [-1, 100, 300, 500, 700, 900, 1100, 2000],
                             labels=["<100", "100-300", "300-500", "500-700", "700-900", "900-1100", ">1100"]),
        "hour": pd.cut(d.hour, [5, 8, 10, 12, 14, 16, 18, 21],
                       labels=["06-08", "08-10", "10-12", "12-14", "14-16", "16-18", "18-21"], right=False),
        "cell_temp": pd.cut(d.t_cell, [-30, 0, 10, 20, 30, 40, 80],
                            labels=["<0", "0-10", "10-20", "20-30", "30-40", ">40"]),
        "wind": pd.cut(d.wind_speed_ms, [-1, 1, 2, 3, 5, 8, 40],
                       labels=["0-1", "1-2", "2-3", "3-5", "5-8", ">8"]),
    }
    rows = []
    for reg, b in bins.items():
        for lab, g in d.groupby(b, observed=True):
            r = {"regime": reg, "bin": str(lab), "n": len(g)}
            for k, c in cols.items():
                r[k] = rmse(g.dc_power_kw.values - g[c].values)
            r["hybrid_vs_physics_pct"] = 100 * (1 - r["hybrid"] / r["physics"])
            r["hybrid_base_vs_physics_pct"] = 100 * (1 - r["hybrid_base"] / r["physics"])
            rows.append(r)
    reg = pd.DataFrame(rows)
    reg.to_csv(FLD / "test2025_regimes.csv", index=False)
    return reg


def fig_regime_lines(reg, which, xlabel, name, title):
    g = reg[reg.regime == which]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    x = np.arange(len(g))
    for m in ["physics", "pure_ml", "linear_res", "hybrid"]:
        lab, c, mk, ls = STYLE[m]
        ax.plot(x, g[m], color=c, marker=mk, ls=ls, ms=4, label=lab)
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(g.bin, g.n)], fontsize=6)
    ax.set_xlabel(xlabel); ax.set_ylabel("RMSE (kW)"); ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, name)


# ======================================================== figures
def fig_sim_groups():
    r = pd.read_csv(SIM / "pvmismatch_results.csv")
    order = ["IID", "T_ood", "G_ood", "shade_ood", "T_ood|shade_ood",
             "G_ood|T_ood", "G_ood|shade_ood", "G_ood|T_ood|shade_ood"]
    names = ["IID", "T-OOD", "G-OOD", "Shade-OOD", "T+Shade", "G+T", "G+Shade", "G+T+Shade"]
    models = ["physics", "linear_res", "pure_ml", "hybrid"]
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    w = 0.2
    x = np.arange(len(order))
    for i, m in enumerate(models):
        s = r[r.model == m].set_index("experiment").loc[order]
        lab, c, _, _ = STYLE[m]
        hatch = {"physics": "///", "linear_res": "..", "pure_ml": "\\\\", "hybrid": ""}[m]
        ax.bar(x + (i - 1.5) * w, s.rmse_w, w * 0.92, color=c, label=lab, hatch=hatch,
               edgecolor="white", linewidth=0.3,
               yerr=[s.rmse_w - s.ci_lo, s.ci_hi - s.rmse_w], error_kw=dict(lw=0.6, capsize=1.5))
    h = r[r.model == "hybrid"].set_index("experiment").loc[order]
    for xi, (v, red) in enumerate(zip(h.ci_hi, h.reduction_vs_physics_pct)):
        ax.text(xi + 1.5 * w, v + 8, f"−{red:.0f}%", ha="center", fontsize=6, color="#0b0b0b")
    ax.set_xticks(x, names)
    ax.set_ylabel("RMSE (W)")
    ax.set_title("PVMismatch 12-panel array: test RMSE by operating regime (95% bootstrap CI)")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_sim_rmse_by_regime")


def fig_sim_severity():
    p = pd.read_csv(SIM / "pvmismatch_predictions_seed42.csv")
    p = p[p.N_shaded > 0]
    b = pd.cut(p.mean_severity, [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.8])
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for m in ["physics", "pure_ml", "linear_res", "hybrid"]:
        lab, c, mk, ls = STYLE[m]
        v = p.groupby(b, observed=True).apply(lambda g: rmse(g.p_reference_w - g[f"pred_{m}"]))
        ax.plot(np.arange(len(v)), v.values, color=c, marker=mk, ls=ls, ms=4, label=lab)
    n = p.groupby(b, observed=True).size()
    ax.set_xticks(np.arange(len(n)), [f"{int(i.left*100)}-{int(i.right*100)}%\n(n={k})" for i, k in n.items()], fontsize=6)
    ax.set_xlabel("Mean array shading severity"); ax.set_ylabel("RMSE (W)")
    ax.set_title("Simulation: error vs shading severity (IID + OOD test)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "fig_sim_rmse_vs_severity")


def fig_sim_parity():
    p = pd.read_csv(SIM / "pvmismatch_predictions_seed42.csv")
    p = p[p.split == "ood"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), sharex=True, sharey=True)
    lim = [0, p.p_reference_w.max() * 1.05]
    for ax, m in zip(axes, ["physics", "pure_ml", "hybrid"]):
        lab, c, _, _ = STYLE[m]
        e = p.p_reference_w - p[f"pred_{m}"]
        ax.scatter(p.p_reference_w, p[f"pred_{m}"], s=3, color=c, alpha=0.35, lw=0)
        ax.plot(lim, lim, color="#0b0b0b", lw=0.7)
        ax.set_title(f"{lab}\nRMSE = {rmse(e):.0f} W")
        ax.set_xlabel("PVMismatch reference P (W)")
        ax.set_aspect("equal"); ax.set_xlim(lim); ax.set_ylim(lim)
    axes[0].set_ylabel("Predicted P (W)")
    fig.suptitle("Simulation OOD test set (all 3,314 OOD rows)", y=1.04, fontsize=8)
    save(fig, "fig_sim_parity_ood")


def fig_field_summary():
    r = pd.read_csv(FLD / "test2025_results.csv").set_index("model")
    rows = [("physics", "Physics"), ("physics+bias", "Physics + offset"),
            ("linear_res [base]", "Linear corr. (base)"), ("pure_ml [base]", "Pure ML (base)"),
            ("hybrid [base]", "Hybrid (base)"),
            ("linear_res [base+time]", "Linear corr. (+time)"), ("pure_ml [base+time]", "Pure ML (+time)"),
            ("hybrid [base+time]", "Hybrid (+time)")]
    col = {"physics": STYLE["physics"][1], "physics+bias": STYLE["physics"][1],
           "linear": STYLE["linear_res"][1], "pure": STYLE["pure_ml"][1], "hybrid": STYLE["hybrid"][1]}
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab) in zip(y, rows):
        s = r.loc[k]
        c = col.get(k, col[k.split(" ")[0].split("_")[0]])
        ax.barh(yi, s.rmse_kw, 0.7, color=c, alpha=1.0 if "time" in k or "physics" == k else 0.55,
                xerr=[[s.rmse_kw - s.ci_lo], [s.ci_hi - s.rmse_kw]], error_kw=dict(lw=0.6, capsize=1.5))
        ax.text(s.ci_hi + 0.6, yi, f"{s.rmse_kw:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.set_xlabel("RMSE on locked 2025 test (kW)")
    ax.set_xlim(40, 64)
    ax.set_title("PVDAQ 2107, train 2024 / test 2025\n(95% day-block bootstrap CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_field_rmse_summary")


def main():
    reg = pvdaq_regimes()
    pd.set_option("display.width", 200)
    print(reg.round(2).to_string(index=False))
    fig_sim_groups()
    fig_sim_severity()
    fig_sim_parity()
    fig_field_summary()
    fig_regime_lines(reg, "hour", "Hour of day (local)", "fig_field_rmse_by_hour",
                     "PVDAQ 2025: RMSE by time of day")
    fig_regime_lines(reg, "irradiance", "POA irradiance (W/m²)", "fig_field_rmse_by_irradiance",
                     "PVDAQ 2025: RMSE by irradiance")


if __name__ == "__main__":
    main()
