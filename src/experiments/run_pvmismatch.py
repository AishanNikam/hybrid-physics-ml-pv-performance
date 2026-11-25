"""
Main simulation experiment on the PVMismatch 12-panel dataset (5000 rows).

Reference power : PVMismatch (cell-level, bypass diodes)       -> p_reference_w
Physics model   : pvlib CEC single-diode, mean-irradiance shade -> p_physics_w

Models compared
  physics      : pvlib prediction as-is
  physics+bias : pvlib + mean training residual (trivial correction baseline)
  linear_res   : pvlib + ridge regression on the residual
  pure_ml      : XGBoost predicting p_reference directly
  hybrid       : pvlib + XGBoost predicting the residual

Protocol
  - Models are trained ONLY on iid rows.
  - IID test: 5 seeds x 80/20 split of iid rows.
  - OOD tests: model trained on all iid rows (5 seeds), evaluated on each
    ood_flag group defined by the dataset generator.
  - 95% CIs: bootstrap over test rows (pooled across seeds), and a paired
    bootstrap for hybrid-vs-pure_ml RMSE difference.

Run from repo root:  python src/experiments/run_pvmismatch.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "pvmismatch_12panel_5000.csv"
OUT = ROOT / "results" / "metrics" / "pvmismatch"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 21, 100, 123]
N_BOOT = 2000
FEATURES = ["G_wm2", "T_amb_c", "N_shaded", "mean_severity", "max_severity"] + [
    f"S{i}" for i in range(1, 13)
]
TARGET, PHYS = "p_reference_w", "p_physics_w"
MODELS = ["physics", "physics+bias", "linear_res", "pure_ml", "hybrid"]

# Optional topology-aware inputs (python run_pvmismatch.py --string-features):
# a series string's current is limited by its weakest panel, so give the model
# per-string minimum / mean sunlight and how many strings contain shade.
STRING_FEATURES = [f"str{k}_min" for k in range(3)] + [f"str{k}_mean" for k in range(3)] + [
    "n_strings_shaded", "min_string_mean", "max_string_mean", "string_spread"]


def add_string_features(df):
    S = df[[f"S{i}" for i in range(1, 13)]].values.reshape(-1, 3, 4)
    smin, smean = S.min(axis=2), S.mean(axis=2)
    for k in range(3):
        df[f"str{k}_min"], df[f"str{k}_mean"] = smin[:, k], smean[:, k]
    df["n_strings_shaded"] = (smin < 1.0).sum(axis=1)
    df["min_string_mean"], df["max_string_mean"] = smean.min(axis=1), smean.max(axis=1)
    df["string_spread"] = df["max_string_mean"] - df["min_string_mean"]
    return df


def xgb(seed):
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", random_state=seed,
        n_jobs=4,
    )


def fit_predict(train, test, seed):
    Xtr, Xte = train[FEATURES], test[FEATURES]
    ytr = train[TARGET].values
    ptr, pte = train[PHYS].values, test[PHYS].values
    res = ytr - ptr
    preds = {"physics": pte, "physics+bias": pte + res.mean()}
    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, res)
    preds["linear_res"] = pte + lin.predict(Xte)
    preds["pure_ml"] = xgb(seed).fit(Xtr, ytr).predict(Xte)
    preds["hybrid"] = pte + xgb(seed).fit(Xtr, res).predict(Xte)
    return preds


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def summarize(name, y, P, seed_rmse, rng):
    """y: pooled targets; P: dict model -> pooled preds; seed_rmse: dict model -> list."""
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rows = []
    for m in MODELS:
        err2 = (y - P[m]) ** 2
        boot = np.sqrt(err2[idx].mean(axis=1))
        rows.append({
            "experiment": name, "model": m, "n_test_per_seed": n // len(SEEDS),
            "rmse_w": rmse(y, P[m]),
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "seed_mean": np.mean(seed_rmse[m]), "seed_std": np.std(seed_rmse[m], ddof=1),
            "mae_w": float(np.mean(np.abs(y - P[m]))),
            "r2": float(1 - err2.sum() / ((y - y.mean()) ** 2).sum()),
            "bias_w": float(np.mean(P[m] - y)),
        })
    phys = rows[0]["rmse_w"]
    for r in rows:
        r["reduction_vs_physics_pct"] = 100 * (1 - r["rmse_w"] / phys)
    # paired bootstrap: hybrid vs pure_ml
    e_h, e_m = (y - P["hybrid"]) ** 2, (y - P["pure_ml"]) ** 2
    d = np.sqrt(e_h[idx].mean(1)) - np.sqrt(e_m[idx].mean(1))
    paired = {
        "experiment": name, "diff_rmse_hybrid_minus_pureml": rmse(y, P["hybrid"]) - rmse(y, P["pure_ml"]),
        "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5),
        "p_hybrid_not_better": float(np.mean(d >= 0)),
    }
    return rows, paired


def main():
    global FEATURES, OUT
    import sys
    df = pd.read_csv(DATA)
    if "--string-features" in sys.argv:
        df = add_string_features(df)
        FEATURES = FEATURES + STRING_FEATURES
        OUT = OUT.parent / "pvmismatch_string"
        OUT.mkdir(parents=True, exist_ok=True)
    iid = df[df.ood_flag == "iid"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    all_rows, all_paired = [], []

    # ---- IID: repeated 80/20 split --------------------------------------
    ys, Ps, seed_r = [], {m: [] for m in MODELS}, {m: [] for m in MODELS}
    for s in SEEDS:
        tr, te = train_test_split(iid, test_size=0.2, random_state=s)
        p = fit_predict(tr, te, s)
        ys.append(te[TARGET].values)
        for m in MODELS:
            Ps[m].append(p[m]); seed_r[m].append(rmse(te[TARGET].values, p[m]))
    r, pr = summarize("IID", np.concatenate(ys), {m: np.concatenate(v) for m, v in Ps.items()}, seed_r, rng)
    all_rows += r; all_paired.append(pr)

    # ---- OOD: train on all iid, test on each flag group ------------------
    groups = [g for g in df.ood_flag.unique() if g != "iid"]
    groups.sort(key=lambda g: (g.count("|"), g))
    preds_by_seed = {s: fit_predict(iid, df[df.ood_flag != "iid"], s) for s in SEEDS}
    ood_all = df[df.ood_flag != "iid"].reset_index(drop=True)
    for g in groups + ["ALL_OOD"]:
        mask = np.ones(len(ood_all), bool) if g == "ALL_OOD" else (ood_all.ood_flag == g).values
        y1 = ood_all.loc[mask, TARGET].values
        y = np.tile(y1, len(SEEDS))
        P = {m: np.concatenate([preds_by_seed[s][m][mask] for s in SEEDS]) for m in MODELS}
        sr = {m: [rmse(y1, preds_by_seed[s][m][mask]) for s in SEEDS] for m in MODELS}
        r, pr = summarize(g, y, P, sr, rng)
        all_rows += r; all_paired.append(pr)

    res = pd.DataFrame(all_rows)
    paired = pd.DataFrame(all_paired)
    res.to_csv(OUT / "pvmismatch_results.csv", index=False)
    paired.to_csv(OUT / "pvmismatch_hybrid_vs_pureml.csv", index=False)

    # Save one IID-seed + full OOD prediction set for plots
    tr, te = train_test_split(iid, test_size=0.2, random_state=42)
    p = fit_predict(tr, te, 42)
    te = te.assign(**{f"pred_{m}": p[m] for m in MODELS}, split="iid_test")
    q = ood_all.assign(**{f"pred_{m}": preds_by_seed[42][m] for m in MODELS}, split="ood")
    pd.concat([te, q]).to_csv(OUT / "pvmismatch_predictions_seed42.csv", index=False)

    pd.set_option("display.width", 200)
    piv = res.pivot(index="experiment", columns="model", values="rmse_w")[MODELS]
    piv.insert(0, "n", res.groupby("experiment").n_test_per_seed.first())
    order = ["IID"] + groups + ["ALL_OOD"]
    print("\nRMSE (W), pooled over 5 seeds")
    print(piv.loc[order].round(1).to_string())
    print("\nHybrid vs pure ML (paired bootstrap, W; negative = hybrid better)")
    print(paired.set_index("experiment").loc[order].round(2).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
