"""
Train the demo models on ALL IID rows of the PVMismatch dataset and save them.
(Same features and XGBoost settings as `run_pvmismatch.py --string-features`, seed 42.)

Run from repo root:  python demo/train_demo_models.py
Creates demo/models/pure_ml.json and demo/models/hybrid_residual.json
"""
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from pv_sim import FEATURES, add_string_features

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "pvmismatch_12panel_5000.csv"
(HERE / "models").mkdir(exist_ok=True)


def xgb():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, objective="reg:squarederror", random_state=42)


df = add_string_features(pd.read_csv(DATA))
iid = df[df.ood_flag == "iid"]
X, y, phys = iid[FEATURES], iid.p_reference_w, iid.p_physics_w

xgb().fit(X, y).save_model(HERE / "models" / "pure_ml.json")
xgb().fit(X, y - phys).save_model(HERE / "models" / "hybrid_residual.json")
print(f"Trained on {len(iid)} IID rows -> saved to {HERE / 'models'}")
