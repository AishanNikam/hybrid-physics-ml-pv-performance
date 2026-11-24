# Hybrid Physics–ML Model for PV Performance Under Partial Shading

**Research project | August–November 2026**

A physics-guided machine-learning study of photovoltaic (PV) power prediction under changing irradiance, temperature, and partial-shading conditions. The project compares a pvlib single-diode physics baseline, a pure machine-learning predictor, and a hybrid residual model. Controlled simulations are used to isolate operating regimes; a chronological holdout from NREL PVDAQ System 2107 is used for external validation on field data.

> **Main real-world result:** On the 2025 PVDAQ holdout (16,837 observations), the hybrid model achieved an RMSE of **56.20 kW**, compared with **59.36 kW** for the physics-only baseline and **56.65 kW** for pure ML. This is a **5.3% RMSE reduction** relative to the physics baseline. The improvement on field data is modest and is reported separately from the much larger gains on the controlled simulation dataset.

---

## Research question

**In which irradiance, temperature, and partial-shading regimes does a physics-based PV model deviate from a higher-fidelity reference or measured power, and can residual machine learning correct part of that error?**

The goal is not to replace the physics model with a black-box predictor. Instead, the hybrid model retains the physics prediction and learns a correction to its residual.

## Method

Three models are compared:

1. **Physics-only:** a pvlib CEC single-diode model provides the baseline power prediction.
2. **Pure ML:** an XGBoost regressor predicts power directly from the input features.
3. **Hybrid residual:** an XGBoost regressor predicts the difference between the reference power and the physics prediction; this correction is added to the physics output.

For the hybrid model:

\[
\Delta P = P_{\mathrm{reference}} - P_{\mathrm{physics}}
\]

\[
P_{\mathrm{hybrid}} = P_{\mathrm{physics}} + f_{\theta}(G, T, s)
\]

where `G` is irradiance, `T` is temperature, and `s` is shading fraction. In the controlled simulation experiments, the target is simulated global maximum power. In the PVDAQ field validation, the target is measured DC power. These are different evaluation settings and should not be conflated.

### Physics baseline

- PV model: pvlib CEC single-diode implementation.
- Module parameterization: Canadian Solar CS6P-260P CEC module parameters.
- Cell temperature: SAPM open-rack glass-glass temperature model.
- Controlled shading representation: effective irradiance is reduced using the supplied shading fraction (`G_eff = G × (1 − s)`).
- Field validation: module-level physics output is scaled to the estimated system size for PVDAQ System 2107.

The single-diode output is treated as a **physics baseline**, not as ground truth. The controlled simulation's global maximum power is the reference target for those experiments; PVDAQ measured DC power is the observed field target.

---

## Datasets

### 1. NREL PVDAQ System 2107 — field data

The cleaned PVDAQ dataset contains **127,309 observations** at 15-minute intervals over the 2017–2025 operating period. Available variables include plane-of-array irradiance, ambient temperature, wind speed, DC power, AC power, and derived operating-condition fields.

The external validation uses a chronological split:

| Split | Period | Samples |
|---|---|---:|
| Train | 2024 | 16,754 |
| Test | 2025 | 16,837 |

This is a temporal holdout: the 2025 observations are not randomly mixed into training. The field dataset does **not** contain explicit partial-shading labels, so this validation assesses naturally varying operating conditions and must not be interpreted as a labelled shading experiment.

### 2. Controlled PV simulation — partial-shading experiments

The processed simulation dataset contains **496 rows**, with:

- Irradiance: 500–1,000 W/m²
- Temperature: 20–50 °C
- Shading fraction: 0–50%
- Simulated global maximum power and physics-model power

**Coverage limitation:** the dataset is not a complete irradiance × temperature × shading factorial grid. The unshaded cases span irradiance levels from 500 to 1,000 W/m², while the 10–50% shading cases are available only at 1,000 W/m². The study therefore does not establish performance across every possible combination of irradiance and shading. No panel-level spatial shading masks are represented in this dataset.

---

## Results

### Controlled simulation: three-model comparison

RMSE values are in watts. Lower is better.

| Experiment | OOD/test definition | Test samples | Physics-only | Pure ML | Hybrid residual | Hybrid RMSE reduction vs physics |
|---|---|---:|---:|---:|---:|---:|
| IID | Random 20% holdout (`random_state=42`) | 100 | 12.550 | 2.760 | **0.733** | 94.2% |
| Temperature OOD | Train: 20–40 °C; test: 41–50 °C | 160 | 11.824 | 5.741 | **0.248** | 97.9% |
| Irradiance OOD | Unshaded only; train: 500–900 W/m²; test: 950–1,000 W/m² | 62 | 20.865 | 20.529 | **6.197** | 70.3% |
| Shading OOD | 1,000 W/m²; train: 0–30% shading; test: 40–50% | 62 | 19.345 | 48.093 | **13.450** | 30.5% |
| Combined OOD | 1,000 W/m²; test: 40–50% shading and 41–50 °C | 20 | 18.779 | 46.150 | **13.040** | 30.6% |

RMSE values are in watts. OOD denotes an out-of-distribution holdout under the definitions shown in the table. The combined OOD test contains only 20 samples and should be read as a small stress test, not as a high-powered statistical benchmark. Pure ML shows substantial extrapolation error in the shading and combined OOD tests; the hybrid model has lower RMSE than the physics baseline in each reported simulation split.

A separate five-seed IID robustness run reported mean RMSE (mean ± standard deviation) of **12.166 ± 0.387 W** for physics-only, **2.213 ± 0.380 W** for pure ML, and **0.556 ± 0.125 W** for the hybrid model. These multi-seed results are reported separately from the single fixed-split table above.

### PVDAQ: chronological 2025 external validation

RMSE and MAE are in kilowatts.

| Model | MAE (kW) | RMSE (kW) | R² | Bias (kW) |
|---|---:|---:|---:|---:|
| Physics-only | 41.90 | 59.36 | 0.9383 | -26.78 |
| Pure ML | 39.40 | 56.65 | 0.9438 | -8.07 |
| Hybrid residual | **39.20** | **56.20** | **0.9447** | -8.60 |

On this real-world holdout, the hybrid model reduces RMSE by **5.3%** relative to physics-only and by **0.8%** relative to pure ML. The physics baseline has a negative mean bias; both ML-based approaches reduce its magnitude. The gain is materially smaller than in the controlled simulation experiments, which is an important result rather than a result to hide.

### PVDAQ error-regime analysis

The regime analysis compares hybrid RMSE with physics-only RMSE within bins of irradiance, cell temperature, wind speed, and time of day. It indicates stronger improvements in some moderate-to-high operating regimes—for example, 9.1% at 700–900 W/m², 8.3% above 40 °C cell temperature, 9.7% at 3–5 m/s wind speed, and 12.6% from 08:00–10:00. The hybrid model is worse than the physics baseline in some low-signal or late-day bins. These are empirical associations; they do not establish the physical cause of the errors. Very small bins (for example, 19 samples above 1,100 W/m² or 69 samples above 8 m/s) should be interpreted cautiously.

---

## Visual results

The following figures are generated by the analysis scripts and are stored under `results/figures/` when those outputs are present in the checkout.

### PVDAQ external validation

- `pvdaq_hybrid_improvement_by_regime.png` — hybrid RMSE reduction relative to physics across all PVDAQ regimes.
- `pvdaq_rmse_irradiance.png` — model RMSE by irradiance bin.
- `pvdaq_rmse_temperature.png` — model RMSE by cell-temperature bin.
- `pvdaq_rmse_wind.png` — model RMSE by wind-speed bin.
- `pvdaq_rmse_hour.png` — model RMSE by time-of-day bin.

### Earlier dataset and simulation analysis

- `fig1_dataset_overview.png`
- `fig2_cleaning_waterfall.png`
- `fig3_pvdaq_distributions.png`
- `fig4_residual_analysis.png`
- `fig5_error_heatmaps.png`
- `fig6_simulation_shading.png`
- `fig7_cross_dataset.png`
- `fig8_long_term_trends.png`
- `fig9_summary_metrics.png`

---

## Repository layout

```text
hybrid-physics-ml-pv-performance/
├── data/
│   ├── processed/
│   │   ├── PVDAQ_2107_clean.csv
│   │   ├── PVDAQ_residuals_clean.csv
│   │   └── PV_simulation_clean.csv
│   └── sample/
│       └── PVDAQ_2107_sample.csv
├── docs/
│   ├── Physics-guided-ML-for-PV-under-partial-shading.pptx
│   ├── PV_Hybrid_ML_Research_Project_Roadmap.pdf
│   ├── PV_Midsem_v2 (1).pptx
│   └── PV_Working_Paper.docx
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── preprocessing/
│   ├── experiments/
│   └── explainability/
├── requirements.txt
└── README.md
```

Large source or processed datasets may be excluded from version control depending on repository file-size limits. Check the actual tracked files before assuming every local data file is available to a fresh clone.

---

## Reproducing the main analyses

Run commands from the repository root after installing the packages in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt

# Fixed-split controlled IID comparison
python src/experiments/run_iid.py

# Multi-seed IID robustness
python src/experiments/run_multiseed.py

# Controlled feature ablation
python src/experiments/run_ablation.py

# Chronological PVDAQ external validation
python src/experiments/run_pvdaq_validation.py

# PVDAQ regime analysis and plots
python src/experiments/pvdaq_error_regime_analysis.py
python src/experiments/plot_pvdaq_analysis.py
```

Additional experiment and plotting scripts are available under `src/experiments/`. Output metrics are written to `results/metrics/`, prediction tables to `results/predictions/`, and figures to `results/figures/`.

---

## Limitations and interpretation

- The controlled simulation dataset is relatively small (496 rows) and its shading/irradiance coverage is unbalanced. OOD results should be interpreted within the stated split design.
- The combined OOD test contains only 20 observations.
- The simulation uses a simplified effective-irradiance shading representation; it does not encode panel-level spatial shading masks or a full range of module/sub-string mismatch patterns.
- PVDAQ System 2107 has no explicit shading labels. Its field validation cannot isolate partial shading as the cause of an observed residual.
- The field-validation improvement is modest (5.3% RMSE reduction over physics-only), despite much larger controlled-simulation gains. This gap motivates further validation across additional systems and operating periods.
- The estimated module scaling and representative module parameterization introduce modelling assumptions in the system-level PVDAQ physics baseline.

## Technology stack

Python, pvlib, XGBoost, scikit-learn, pandas, NumPy, Matplotlib.

## Authors

Teghvir Singh Jassal, Anshu, and Aishaan.

## Citation

A formal publication citation is not available yet. Please cite this repository as a research software project and include the repository URL and access date if referencing the current implementation.
