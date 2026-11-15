"""Shared helpers used by the analysis scripts."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def coef_table(results, decimals=4):
    """Return a tidy coefficient table from a fitted statsmodels OLS result."""
    table = pd.DataFrame({
        "coef": results.params,
        "std_err": results.bse,
        "t": results.tvalues,
        "p_value": results.pvalues,
    })
    return table.round(decimals)


def section(title):
    """Print a section header so the console output is easy to read."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
