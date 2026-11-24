# Hybrid Physics–ML Model for PV Performance Under Partial Shading

**Research project | August–November 2026**

A physics-guided machine-learning study investigating photovoltaic (PV) power prediction under changing irradiance, temperature, and partial-shading conditions.

The project compares three approaches:

1. A physics-based PV model using the pvlib CEC single-diode model.

2. A pure machine-learning model using XGBoost.

3. A hybrid physics–ML model that learns to correct the residual errors of the physics model.

Controlled PV simulations are used to investigate performance under defined operating conditions and out-of-distribution (OOD) regimes. A chronological holdout evaluation using NREL PVDAQ System 2107 assesses performance on real-world field measurements.

> **Main real-world result:** On the 2025 PVDAQ chronological holdout (16,837 observations), the hybrid model achieved an RMSE of **56.12 kW**, compared with **59.36 kW** for the physics-only baseline and **56.41 kW** for pure ML. This represents a **5.46% RMSE reduction** relative to physics-only. The improvement over pure ML was 0.51%. The relatively modest field-data improvement is reported separately from the larger gains observed in controlled simulations.

---

## 1. Research Question

**In which irradiance, temperature, and partial-shading regimes does a physics-based PV model deviate from a higher-fidelity reference or measured power, and can residual machine learning correct part of that error?**

The objective is not to replace the physics model with a black-box ML predictor. Instead, the hybrid approach retains the physical model prediction and learns a correction to its residual error.

## 2. Methodology

Three models are evaluated.

### 2.1 Physics-only model

A physics-based photovoltaic model provides the baseline power prediction:

\\\[

P\_{\\text{physics}}

\\]

The model is implemented using pvlib and a CEC single-diode modelling approach.

### 2.2 Pure machine-learning model

An XGBoost regressor directly predicts PV power from the input features:

\\\[

X \\rightarrow P\_{\\text{ML}}

\\]

For controlled simulations, the input features are:

- Irradiance

- Temperature

- Shading fraction

For PVDAQ, the input features are:

- Plane-of-array irradiance

- Ambient temperature

- Estimated cell temperature

- Wind speed

### 2.3 Hybrid residual model

The hybrid model learns the difference between the reference power and the physics prediction:

\\\[

\\Delta P = P\_{\\text{reference}} - P\_{\\text{physics}}

\\]

The ML model estimates this residual:

\\\[

\\Delta P\_{\\text{ML}} = f\_\\theta(X)

\\]

The final hybrid prediction is:

\\\[

P\_{\\text{hybrid}}

=

P\_{\\text{physics}}+\\Delta P\_{\\text{ML}}

\\]

In controlled simulations, the reference target is simulated global maximum power. In PVDAQ, the reference target is measured DC power.

These are different evaluation settings and their results should not be conflated.

### 2.4 Physics model configuration

The physics baseline uses:

- **PV model:** pvlib CEC single-diode implementation.

- **Module parameterization:** Canadian Solar CS6P-260P CEC module parameters.

- **Cell temperature:** SAPM open-rack glass-glass temperature model.

- **Controlled shading representation:** effective irradiance is reduced using the supplied shading fraction:

\\\[

G\_{\\text{effective}}=G(1-s)

\\]

- **PVDAQ system scaling:** module-level physics predictions are scaled using an estimated module count of 2,797, corresponding to an estimated system STC capacity of approximately 727.85 kW.

The single-diode output is treated as a physics baseline, not as ground truth. The controlled simulation's global maximum power is the reference target for simulation experiments, while PVDAQ measured DC power is the observed field target.

### 2.5 Machine-learning configuration

The main XGBoost configuration uses:

| Parameter | Value |

|---|---:|

| Number of estimators | 300 |

| Maximum tree depth | 4 |

| Learning rate | 0.03 |

| Subsample | 0.8 |

| Column subsample | 0.8 |

| Objective | Squared-error regression |

| Random state | 42 |

The residual model follows the same general configuration, with the random seed varied in the multi-seed experiment.

---

## 3. Datasets

The project uses two complementary data sources.

### 3.1 NREL PVDAQ System 2107 — Real-world field data

The cleaned PVDAQ System 2107 dataset contains:

- 127,309 observations

- 15-minute measurement intervals

- Operating period: 2017–2025

- Plane-of-array irradiance

- Ambient temperature

- Wind speed

- DC power

- AC power

- Derived operating-condition variables

#### Chronological holdout evaluation

The field-data evaluation uses a chronological split:

| Split | Period | Samples |

|---|---|---:|

| Training | 2024 | 16,754 |

| Test | 2025 | 16,837 |

The 2025 observations are held out from model training and evaluation tuning.

Missing wind-speed values used in the cell-temperature calculation are filled using the 2024 training-period median wind speed. The same training-derived value is applied to the test period.

This is a chronological holdout evaluation on the same PVDAQ System 2107, not an independent validation on a separate PV installation.

**Important:** PVDAQ System 2107 does not contain explicit partial-shading labels. Its field evaluation therefore assesses naturally varying operating conditions and cannot isolate partial shading as the cause of an observed residual.

### 3.2 Controlled PV simulation — Partial-shading experiments

The processed controlled simulation dataset contains **496 observations**.

Its main variables include:

- Irradiance: 500–1,000 W/m²

- Temperature: 20–50 °C

- Shading fraction: 0–50%

- Simulated global maximum power

- Local maximum power, where available

- Physics-model power

- Physics residual

- Peak mismatch

#### Simulation coverage limitation

The dataset is not a complete irradiance × temperature × shading factorial grid.

- Unshaded observations span irradiance levels from 500 to 1,000 W/m².

- Shading levels of 10–50% are available only at 1,000 W/m².

- The dataset does not represent every possible combination of irradiance, temperature, and shading.

- No panel-level spatial shading masks are represented.

Consequently, the OOD results should be interpreted within the specific operating conditions covered by the dataset.

### 3.3 Data availability

The full cleaned dataset `PVDAQ\_2107\_clean.csv` is intentionally excluded from Git using `.gitignore`.

The repository tracks:

```text

data/

├── processed/

│   ├── PVDAQ\_residuals\_clean.csv

│   └── PV\_simulation\_clean.csv

│

└── sample/

&#x20;   └── PVDAQ\_2107\_sample.csv

```

The sample dataset is not sufficient to reproduce the complete 2024–2025 PVDAQ evaluation.

To rerun the complete PVDAQ validation, the corresponding full NREL PVDAQ System 2107 dataset must be obtained and processed into:

```text

data/processed/PVDAQ\_2107\_clean.csv

```

Researchers reproducing the field evaluation should consult NREL PVDAQ for the source data and follow its data-access and attribution requirements.

---

## 4. Experimental Design

The project evaluates performance under five controlled simulation experiments.

### 4.1 IID experiment

A random 80/20 train-test split is created using:

```python

test\_size=0.20

random\_state=42

```

The split contains:

- Training: 396 observations

- Testing: 100 observations

This evaluates performance on a random holdout from the sampled simulation conditions. It tests interpolation within the dataset rather than generalization to entirely new physical regimes.

### 4.2 Temperature OOD

| Dataset portion | Temperature |

|---|---|

| Training | 20–40 °C |

| Testing | 41–50 °C |

The test set contains temperature values beyond the training range.

### 4.3 Irradiance OOD

This experiment uses only unshaded observations to isolate irradiance extrapolation.

| Dataset portion | Irradiance | Shading |

|---|---|---|

| Training | 500–900 W/m² | 0% |

| Testing | 950–1,000 W/m² | 0% |

### 4.4 Shading OOD

Irradiance is fixed at 1,000 W/m².

| Dataset portion | Shading |

|---|---|

| Training | 0–30% |

| Testing | 40–50% |

This evaluates performance under shading levels not represented in training.

### 4.5 Combined OOD stress test

This experiment evaluates a joint severe-shading and high-temperature condition.

| Dataset portion | Irradiance | Shading | Temperature |

|---|---|---|---|

| Training | 1,000 W/m² | 0–30% | 20–50 °C |

| Testing | 1,000 W/m² | 40–50% | 41–50 °C |

The test set contains 20 observations.

**Important interpretation:** Temperatures of 41–50 °C are represented in the training dataset under lower shading conditions. Therefore, temperature is not individually unseen. The held-out condition is the combination of severe shading with high temperature.

This experiment should be interpreted as a small joint-condition stress test, not as a statistically comprehensive benchmark.

---

## 5. Experimental Results

### 5.1 Controlled simulation: Three-model comparison

The following table compares the physics-only, pure ML, and hybrid residual models.

All RMSE values are in watts. Lower RMSE indicates better prediction accuracy.

| Experiment | Test samples | Physics RMSE | Pure ML RMSE | Hybrid RMSE | Hybrid reduction vs Physics |

|---|---:|---:|---:|---:|---:|

| IID | 100 | 12.550 | 2.760 | **0.733** | 94.2% |

| Temperature OOD | 160 | 11.824 | 5.741 | **0.248** | 97.9% |

| Irradiance OOD | 62 | 20.865 | 20.529 | **6.197** | 70.3% |

| Shading OOD | 62 | 19.345 | 48.093 | **13.450** | 30.5% |

| Combined OOD | 20 | 18.779 | 46.150 | **13.040** | 30.6% |

Across all five evaluated simulation regimes, the hybrid residual model achieves the lowest RMSE.

The largest reductions relative to the physics baseline occur in the IID and temperature OOD experiments.

Under severe unseen shading, pure ML shows substantial extrapolation error, while the hybrid model retains a lower RMSE than both the pure ML model and the physics baseline.

The combined OOD result should be interpreted cautiously because its test set contains only 20 observations.

### 5.2 Multi-seed IID robustness

A separate five-seed repeated random holdout experiment evaluates sensitivity to the random train-test split.

Seeds:

```text

\[42, 7, 21, 100, 123]

```

Mean RMSE and sample standard deviation:

| Model | Mean RMSE (W) | Standard deviation (W) |

|---|---:|---:|

| Physics-only | 12.166 | 0.387 |

| Pure ML | 2.213 | 0.380 |

| Hybrid residual | **0.556** | **0.125** |

The hybrid model achieves the lowest mean RMSE and the lowest variability across the five evaluated seeds.

This experiment is a repeated random holdout analysis, not five-fold cross-validation.

### 5.3 Feature ablation study

The ablation experiment evaluates seven feature combinations:

- Irradiance + Temperature + Shading

- Irradiance + Temperature

- Irradiance + Shading

- Temperature + Shading

- Irradiance only

- Temperature only

- Shading only

The ablation results indicate:

- In IID and temperature OOD experiments, the full three-feature combination achieves the lowest RMSE.

- In irradiance OOD, irradiance alone performs marginally better, consistent with the experiment isolating unshaded irradiance variation.

- In shading OOD, shading alone performs marginally better.

- In combined OOD, shading alone also performs marginally better.

These findings describe performance under the specific experimental splits. They should not be interpreted as universal feature-importance rankings or causal effects.

The full-feature IID ablation uses the same 396/100 split as the main IID experiment.

---

## 6. PVDAQ Chronological Holdout Results

The PVDAQ evaluation uses 2024 for training and 2025 for testing.

All MAE, RMSE, and bias values are in kilowatts.

| Model | MAE (kW) | RMSE (kW) | R² | Bias (kW) |

|---|---:|---:|---:|---:|

| Physics-only | 41.90 | 59.36 | 0.9383 | -26.78 |

| Pure ML | 39.24 | 56.41 | 0.9443 | -7.88 |

| Hybrid residual | **39.13** | **56.12** | **0.9449** | -8.50 |

The hybrid residual model reduces RMSE by:

- **5.46% relative to physics-only**

- **0.51% relative to pure ML**

The physics baseline exhibits negative mean bias. Both ML-based approaches reduce the magnitude of this bias.

The improvement on field data is substantially smaller than the gains observed in controlled simulations.

This difference is an important finding: strong performance on a controlled simulation dataset does not automatically translate into equally large improvements on real-world PV data.

The field evaluation is limited to a chronological holdout from one PV system. Further validation on additional PV installations and operating periods is necessary.

---

## 7. PVDAQ Error-Regime Analysis

The PVDAQ error-regime analysis compares hybrid RMSE with physics-only RMSE across operating-condition bins.

The evaluated regimes are:

- Irradiance

- Estimated cell temperature

- Wind speed

- Time of day

Selected findings from the corrected 2025 holdout:

| Operating regime | Hybrid RMSE reduction vs Physics |

|---|---:|

| Irradiance 700–900 W/m² | 9.44% |

| Irradiance 900–1,100 W/m² | 9.34% |

| Cell temperature above 40 °C | 8.65% |

| Wind speed 2–3 m/s | 8.47% |

| Wind speed 3–5 m/s | 9.76% |

| Time 08:00–10:00 | 12.58% |

| Time 10:00–12:00 | 10.45% |

The hybrid model is worse than the physics baseline in some regimes, including:

- Irradiance below 100 W/m²

- Wind speed above 8 m/s

- Late-afternoon and evening periods

For example, the 16:00–18:00 regime shows a 14.37% increase in RMSE relative to physics-only.

These are empirical associations. They do not establish the physical cause of the prediction errors. Small sample groups, such as the 19 observations above 1,100 W/m², should be interpreted cautiously.

Since the PVDAQ dataset has no explicit shading labels, these field-error patterns cannot be attributed specifically to partial shading.

---

## 8. Visual Results

Figures generated by the analysis scripts are stored in `results/figures/`.

### PVDAQ chronological holdout evaluation

| Figure | Description |

|---|---|

| `pvdaq\_hybrid\_improvement\_by\_regime.png` | Hybrid RMSE reduction relative to physics across PVDAQ regimes |

| `pvdaq\_rmse\_irradiance.png` | Model RMSE by irradiance bin |

| `pvdaq\_rmse\_temperature.png` | Model RMSE by cell-temperature bin |

| `pvdaq\_rmse\_wind.png` | Model RMSE by wind-speed bin |

| `pvdaq\_rmse\_hour.png` | Model RMSE by time-of-day bin |

### Earlier dataset and simulation analysis

- `fig1\_dataset\_overview.png`

- `fig2\_cleaning\_waterfall.png`

- `fig3\_pvdaq\_distributions.png`

- `fig4\_residual\_analysis.png`

- `fig5\_error\_heatmaps.png`

- `fig6\_simulation\_shading.png`

- `fig7\_cross\_dataset.png`

- `fig8\_long\_term\_trends.png`

- `fig9\_summary\_metrics.png`

Other prediction and comparison plots are also available in `results/figures/`.

---

## 9. Repository Structure

```text

hybrid-physics-ml-pv-performance/

│

├── data/

│   ├── processed/

│   │   ├── PVDAQ\_2107\_clean.csv          # Local only; ignored by Git

│   │   ├── PVDAQ\_residuals\_clean.csv

│   │   └── PV\_simulation\_clean.csv

│   │

│   └── sample/

│       └── PVDAQ\_2107\_sample.csv

│

├── docs/

│   ├── Physics-guided-ML-for-PV-under-partial-shading.pptx

│   ├── PV\_Hybrid\_ML\_Research\_Project\_Roadmap.pdf

│   ├── PV\_Midsem\_v2 (1).pptx

│   └── PV\_Working\_Paper.docx

│

├── results/

│   ├── figures/

│   ├── metrics/

│   └── predictions/

│

├── src/

│   ├── preprocessing/

│   │   └── exploratory\_analysis.py

│   │

│   ├── experiments/

│   │   ├── audit\_dataset.py

│   │   ├── error\_regime\_analysis.py

│   │   ├── final\_comparison.py

│   │   ├── run\_iid.py

│   │   ├── run\_temperature\_ood.py

│   │   ├── run\_irradiance\_ood.py

│   │   ├── run\_shading\_ood.py

│   │   ├── run\_combined\_ood.py

│   │   ├── run\_ablation.py

│   │   ├── run\_multiseed.py

│   │   ├── run\_pvdaq\_validation.py

│   │   ├── pvdaq\_error\_regime\_analysis.py

│   │   ├── plot\_pvdaq\_analysis.py

│   │   ├── save\_predictions.py

│   │   ├── plot\_predictions.py

│   │   ├── plot\_hybrid\_improvement.py

│   │   └── additional plotting scripts

│   │

│   └── explainability/

│       └── feature\_importance.py

│

├── requirements.txt

├── .gitignore

└── README.md

```

**Note:** `PVDAQ\_2107\_clean.csv` is required for rerunning the full PVDAQ validation but is not included in the Git repository. The tracked sample file does not replace the full dataset.

---

## 10. Reproducibility

### 10.1 Environment

The project was developed and evaluated using:

- Python 3.13.5

- NumPy 2.4.0

- Pandas 2.3.3

- Matplotlib 3.10.8

- Seaborn 0.13.2

- SciPy 1.18.1

- Scikit-learn 1.9.1

- XGBoost 3.4.1

- pvlib 0.16.1

Install the dependencies from the repository root:

```powershell

python -m pip install -r requirements.txt

```

### 10.2 Run the controlled experiments

```powershell

# Fixed-split IID experiment

python src/experiments/run\_iid.py

# Temperature OOD

python src/experiments/run\_temperature\_ood.py

# Irradiance OOD

python src/experiments/run\_irradiance\_ood.py

# Shading OOD

python src/experiments/run\_shading\_ood.py

# Combined OOD stress test

python src/experiments/run\_combined\_ood.py

# Feature ablation

python src/experiments/run\_ablation.py

# Five-seed IID robustness

python src/experiments/run\_multiseed.py

# Final controlled-simulation comparison

python src/experiments/final\_comparison.py

```

### 10.3 Run the PVDAQ evaluation

The full cleaned PVDAQ file must first be available at:

```text

data/processed/PVDAQ\_2107\_clean.csv

```

Then run:

```powershell

# PVDAQ chronological holdout evaluation

python src/experiments/run\_pvdaq\_validation.py

# PVDAQ error-regime analysis

python src/experiments/pvdaq\_error\_regime\_analysis.py

# Generate PVDAQ regime figures

python src/experiments/plot\_pvdaq\_analysis.py

```

The evaluation writes metrics to `results/metrics/`, predictions to `results/predictions/`, and figures to `results/figures/`.

Run all commands from the repository root unless a script explicitly specifies otherwise.

---

## 11. Limitations and Interpretation

The following limitations apply to the current study:

1. **Simulation coverage:** The controlled simulation dataset contains 496 rows and has unbalanced irradiance/shading coverage. Results apply to the tested regimes and do not establish performance across a complete factorial grid.

2. **Small combined OOD sample:** The combined shading–temperature stress test contains only 20 observations. It should not be treated as a high-powered statistical benchmark.

3. **Simplified shading representation:** Shading is represented through effective irradiance reduction. The simulation does not encode panel-level spatial shading masks or the full range of module and substring mismatch patterns.

4. **PVDAQ shading labels:** System 2107 has no explicit shading labels. The real-world evaluation cannot isolate partial shading as the cause of observed residuals.

5. **Single-system field evaluation:** The PVDAQ results represent a chronological holdout on one PV installation. Independent validation across additional systems is required.

6. **Modest field improvement:** The hybrid model reduces PVDAQ RMSE by 5.46% relative to the physics baseline, substantially less than the improvement observed in controlled simulations. This gap is an important limitation and motivates further investigation.

7. **Physics model assumptions:** The representative module parameterization and estimated system scaling introduce modelling assumptions into the system-level physics baseline.

8. **Regime interpretation:** Error-regime comparisons identify empirical associations, not causal physical explanations. Results from small bins should be interpreted cautiously.

9. **Data availability:** The complete cleaned PVDAQ dataset is excluded from version control. A fresh repository clone cannot reproduce the complete PVDAQ evaluation without separately obtaining and preparing the full data.

---

## 12. Technology Stack

- **Language:** Python

- **Physics modelling:** pvlib

- **Machine learning:** XGBoost

- **Evaluation and model utilities:** Scikit-learn

- **Data processing:** Pandas, NumPy

- **Visualisation:** Matplotlib, Seaborn

- **Scientific computing:** SciPy

---

## 13. Authors

- Teghvir Singh Jassal

- Anshu Kumar

- Aishaan

---

## 14. Research Status

This repository contains the implementation, controlled simulation experiments, PVDAQ chronological holdout evaluation, error-regime analysis, and associated visual results for the research project.

A formal publication citation is not yet available. When referencing the current implementation, cite the repository URL and access date.

**GitHub repository:**

https://github.com/teghvirsingh2006/hybrid-physics-ml-pv-performance
