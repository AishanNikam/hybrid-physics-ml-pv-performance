"""
PVDAQ System 2107 field evaluation (v2).

Data protocol (no test leakage)
  - Only 2024-2025 have measured DC power (2018-23 is inverter-derived),
    so the target is consistent: dc_power_kw.
  - 2024 = development year. All choices (feature set) are made by
    blocked cross-validation INSIDE 2024 (6 folds of 2 months each).
  - 2025 = locked test year, evaluated once for every model.
  - 95% CIs on 2025 use a day-block bootstrap (readings within a day are
    autocorrelated), paired across models.

Models
  physics, physics+bias, linear_res, pure_ml, hybrid  (as in run_pvmismatch.py)
  each ML model with feature set "base" (original 4 inputs) and
  "base+time" (adds cyclic hour-of-day and day-of-year, a proxy for sun
  position since site coordinates are not in the dataset).
  Predictions are clipped at 0 kW (a PV array cannot produce negative DC power).

Run from repo root:  python src/experiments/run_pvdaq_v2.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pvlib.pvsystem import calcparams_cec, retrieve_sam, singlediode
from pvlib.temperature import sapm_cell
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "PVDAQ_2107_clean.csv"
OUT = ROOT / "results" / "metrics" / "pvdaq_v2"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "dc_power_kw"
SEEDS = [42, 7, 21, 100, 123]
N_BOOT = 2000
FEATSETS = {
    "base": ["poa_irradiance_wm2", "ambient_temperature_c", "t_cell", "wind_speed_ms"],
}
FEATSETS["base+time"] = FEATSETS["base"] + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
ML_MODELS = ["linear_res", "pure_ml", "hybrid"]


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df[df.timestamp.dt.year.isin([2024, 2025])]
    df = df.dropna(subset=[TARGET, "poa_irradiance_wm2", "ambient_temperature_c"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    dev = df.timestamp.dt.year == 2024
    wind_med = df.loc[dev, "wind_speed_ms"].median()          # from 2024 only
    df["wind_speed_ms"] = df["wind_speed_ms"].fillna(wind_med)
    df["t_cell"] = sapm_cell(df.poa_irradiance_wm2.values, df.ambient_temperature_c.values,
                             df.wind_speed_ms.values, a=-3.47, b=-0.0594, deltaT=3)
    h = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    doy = df.timestamp.dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["day"] = df.timestamp.dt.date

    # physics baseline: CEC single-diode, CS6P-260P, scaled by module count
    mod = retrieve_sam("CECMod")["Canadian_Solar_Inc__CS6P_260P"]
    n_mod = int(np.ceil(df.loc[dev, TARGET].max() * 1000 / float(mod["STC"])))  # 2024 only
    IL, I0, Rs, Rsh, nVth = calcparams_cec(
        np.maximum(df.poa_irradiance_wm2.values, 0), df.t_cell.values,
        mod["alpha_sc"], mod["a_ref"], mod["I_L_ref"], mod["I_o_ref"],
        mod["R_sh_ref"], mod["R_s"], mod["Adjust"])
    pmp = np.maximum(np.asarray(singlediode(IL, I0, Rs, Rsh, nVth)["p_mp"], float), 0)
    df["physics_kw"] = np.nan_to_num(pmp) * n_mod / 1000
    return df, n_mod


# -------------------------------------------------------------- models
def xgb(seed):
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, objective="reg:squarederror",
                        random_state=seed, n_jobs=4)


def fit_predict(tr, te, feats, seed):
    Xtr, Xte = tr[feats], te[feats]
    ytr, ptr, pte = tr[TARGET].values, tr.physics_kw.values, te.physics_kw.values
    res = ytr - ptr
    out = {}
    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, res)
    out["linear_res"] = pte + lin.predict(Xte)
    out["pure_ml"] = xgb(seed).fit(Xtr, ytr).predict(Xte)
    out["hybrid"] = pte + xgb(seed).fit(Xtr, res).predict(Xte)
    return {k: np.maximum(v, 0) for k, v in out.items()}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# ------------------------------------------------- step 1: CV inside 2024
def dev_cv(dev):
    folds = (dev.timestamp.dt.month - 1) // 2          # 6 blocks of 2 months
    rows = []
    for fs, feats in FEATSETS.items():
        for k in range(6):
            tr, va = dev[folds != k], dev[folds == k]
            p = fit_predict(tr, va, feats, 42)
            y = va[TARGET].values
            rows.append({"featset": fs, "fold": k, "physics": rmse(y, va.physics_kw.values),
                         **{m: rmse(y, p[m]) for m in ML_MODELS}})
    cv = pd.DataFrame(rows)
    summary = cv.groupby("featset")[["physics"] + ML_MODELS].mean()
    return cv, summary


# --------------------------------------------- step 2: locked 2025 test
def day_block_boot(test, preds, rng):
    days = test.day.values
    uniq, inv = np.unique(days, return_inverse=True)
    y = test[TARGET].values
    # per-day sums of squared error and counts -> fast block bootstrap
    cnt = np.bincount(inv)
    sse = {m: np.bincount(inv, (y - p) ** 2) for m, p in preds.items()}
    draw = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    n = cnt[draw].sum(1)
    return {m: np.sqrt(s[draw].sum(1) / n) for m, s in sse.items()}


def main():
    df, n_mod = load()
    dev = df[df.timestamp.dt.year == 2024].reset_index(drop=True)
    test = df[df.timestamp.dt.year == 2025].reset_index(drop=True)
    print(f"Dev 2024: {len(dev)} rows | Locked test 2025: {len(test)} rows | modules: {n_mod}")

    cv, cvsum = dev_cv(dev)
    cv.to_csv(OUT / "dev2024_blocked_cv.csv", index=False)
    chosen = cvsum["hybrid"].idxmin()
    print("\nStep 1 - blocked CV inside 2024 (mean RMSE kW over 6 folds):")
    print(cvsum.round(2).to_string())
    print(f"-> feature set chosen on 2024 only: {chosen}")

    # Final: train on all 2024, evaluate once on 2025 (all configs reported)
    y = test[TARGET].values
    preds = {"physics": test.physics_kw.values,
             "physics+bias": np.maximum(test.physics_kw.values + (dev[TARGET] - dev.physics_kw).mean(), 0)}
    seed_rmse = {}
    for fs, feats in FEATSETS.items():
        per_seed = [fit_predict(dev, test, feats, s) for s in SEEDS]
        for m in ML_MODELS:
            key = f"{m} [{fs}]"
            preds[key] = np.mean([p[m] for p in per_seed], axis=0)      # seed-averaged
            seed_rmse[key] = [rmse(y, p[m]) for p in per_seed]

    rng = np.random.default_rng(0)
    boot = day_block_boot(test, preds, rng)
    phys = rmse(y, preds["physics"])
    rows = []
    for m, p in preds.items():
        sr = seed_rmse.get(m, [rmse(y, p)])
        rows.append({"model": m, "rmse_kw": rmse(y, p),
                     "ci_lo": np.percentile(boot[m], 2.5), "ci_hi": np.percentile(boot[m], 97.5),
                     "seed_std": np.std(sr, ddof=1) if len(sr) > 1 else 0.0,
                     "mae_kw": float(np.mean(np.abs(y - p))), "bias_kw": float(np.mean(p - y)),
                     "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                     "reduction_vs_physics_pct": 100 * (1 - rmse(y, p) / phys),
                     "chosen_on_dev": m.endswith(f"[{chosen}]")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test2025_results.csv", index=False)

    pairs = []
    for fs in FEATSETS:
        for a, b in [(f"hybrid [{fs}]", f"pure_ml [{fs}]"), (f"hybrid [{fs}]", "physics")]:
            d = boot[a] - boot[b]
            pairs.append({"a": a, "b": b, "rmse_diff_kw": rmse(y, preds[a]) - rmse(y, preds[b]),
                          "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5),
                          "p_a_not_better": float(np.mean(d >= 0))})
    pairs.append({"a": "hybrid [base+time]", "b": "hybrid [base]",
                  "rmse_diff_kw": rmse(y, preds["hybrid [base+time]"]) - rmse(y, preds["hybrid [base]"]),
                  **dict(zip(["ci_lo", "ci_hi"], np.percentile(boot["hybrid [base+time]"] - boot["hybrid [base]"], [2.5, 97.5]))),
                  "p_a_not_better": float(np.mean(boot["hybrid [base+time]"] >= boot["hybrid [base]"]))})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "test2025_paired.csv", index=False)

    keep = ["timestamp", TARGET, "poa_irradiance_wm2", "t_cell", "wind_speed_ms", "physics_kw"]
    test[keep].assign(**{f"pred_{k}": v for k, v in preds.items()}).to_csv(
        OUT / "test2025_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nStep 2 - locked 2025 test (day-block bootstrap 95% CI):")
    print(res.round(3).to_string(index=False))
    print("\nPaired comparisons (negative diff = first model better):")
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
