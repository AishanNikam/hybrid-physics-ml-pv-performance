import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SIM_PATH = ROOT / "data" / "processed" / "PV_simulation_clean.csv"
PVDAQ_PATH = ROOT / "data" / "processed" / "PVDAQ_residuals_clean.csv"


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def audit_dataset(path, name):

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i:2d}. {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe().T)

    print("\nUnique values:")
    for col in df.columns:
        if df[col].nunique() <= 20:
            print(f"\n{col}:")
            print(df[col].unique())


# --------------------------------------------------
# Run audit
# --------------------------------------------------

audit_dataset(
    SIM_PATH,
    "PV Simulation"
)

audit_dataset(
    PVDAQ_PATH,
    "PVDAQ Residuals"
)
