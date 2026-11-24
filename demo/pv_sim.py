"""
Shared simulation + feature helpers for the demo.

Same physics as notebooks/Sprint2_dataset_generation.ipynb (fixed version):
  - 3 strings x 4 modules, module = AU_Optronics_PM096B00_323 (96 cells)
  - reference : PVMismatch (cell-level, bypass diodes)
  - physics   : pvlib CEC single-diode at the array-mean irradiance
  - both use the same SAPM cell temperature (wind 1 m/s)

Panel numbering: P1..P12, string = (panel-1)//4  ->  P1-P4 string 1, P5-P8 string 2, P9-P12 string 3.
"""
import numpy as np
import pandas as pd
import pvlib
from pvmismatch import PVsystem
from pvmismatch.pvmismatch_lib.pvcell import PVcell
from pvmismatch.pvmismatch_lib.pvmodule import PVmodule

MOD = pvlib.pvsystem.retrieve_sam("CECMod")["AU_Optronics_PM096B00_323"]
ALPHA_ISC = float(MOD["alpha_sc"]) / float(MOD["I_sc_ref"])
TEMP_PARAMS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

FEATURES = ["G_wm2", "T_amb_c", "N_shaded", "mean_severity", "max_severity"] + [
    f"S{i}" for i in range(1, 13)
]


def cell_temp_c(G, T_amb, mask):
    """SAPM cell temperature at the array-mean effective irradiance."""
    G_eff = G * float(np.mean(mask))
    return float(pvlib.temperature.sapm_cell(G_eff, T_amb, 1.0, **TEMP_PARAMS))


def pvmismatch_power(G, T_amb, mask):
    """Reference: PVMismatch array maximum power (W)."""
    mask = np.asarray(mask, float)
    cell = PVcell(alpha_Isc=ALPHA_ISC)
    pvsys = PVsystem(numberStrs=3, numberMods=4, pvmods=PVmodule(pvcells=[cell] * 96))
    pvsys.setTemps(cell_temp_c(G, T_amb, mask) + 273.15)
    Ee = (G / 1000.0) * mask.reshape(3, 4)
    pvsys.setSuns({s: {m: float(Ee[s, m]) for m in range(4)} for s in range(3)})
    return float(pvsys.Pmp)


def pvlib_power(G, T_amb, mask):
    """Physics: pvlib single-diode at the array-mean irradiance, x12 modules (W)."""
    G_eff = G * float(np.mean(mask))
    IL, I0, Rs, Rsh, nVth = pvlib.pvsystem.calcparams_cec(
        G_eff, cell_temp_c(G, T_amb, mask), MOD["alpha_sc"], MOD["a_ref"], MOD["I_L_ref"],
        MOD["I_o_ref"], MOD["R_sh_ref"], MOD["R_s"], MOD["Adjust"],
        EgRef=1.121, dEgdT=-0.0002677)
    return float(pvlib.pvsystem.singlediode(IL, I0, Rs, Rsh, nVth, method="brentq")["p_mp"]) * 12


def features(G, T_amb, mask):
    """One-row DataFrame with exactly the model input columns."""
    mask = np.asarray(mask, float)
    sev = 1.0 - mask
    row = {"G_wm2": G, "T_amb_c": T_amb, "N_shaded": int((mask < 1.0).sum()),
           "mean_severity": float(sev.mean()), "max_severity": float(sev.max())}
    row.update({f"S{i+1}": float(mask[i]) for i in range(12)})
    return pd.DataFrame([row])[FEATURES]
