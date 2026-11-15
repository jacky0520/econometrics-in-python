"""
04 - Asset pricing models for Apple (AAPL)

Part A: CAPM estimated with close prices vs. dividend-adjusted prices.
Part B: CAPM -> add size and value -> profitability (and investment)
        -> momentum, comparing alpha and explanatory power.
        Factors are official Fama-French data or ETF-based proxies,
        depending on how data/download_data.py was run.

Data: run `python data/download_data.py` first.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from helpers import DATA_DIR, FIG_DIR, coef_table, section

DATA_FILE = DATA_DIR / "aapl_factors_monthly.csv"


def load_data():
    if not DATA_FILE.exists():
        sys.exit(f"Data file not found: {DATA_FILE}\n"
                 "Run `python data/download_data.py` first.")

    data = pd.read_csv(DATA_FILE, parse_dates=["date"])

    # Monthly simple returns in percent (same units as the Fama-French factors)
    data["ret_close"] = data["close"].pct_change() * 100
    data["ret_adj"] = data["adj_close"].pct_change() * 100

    # Excess returns over the risk-free rate
    data["exret_close"] = data["ret_close"] - data["RF"]
    data["exret_adj"] = data["ret_adj"] - data["RF"]

    return data.dropna().reset_index(drop=True)


def capm_close_vs_adjusted(data):
    section("Part A: CAPM with unadjusted vs. adjusted prices")
    print(data[["MKT_RF", "exret_close", "exret_adj"]].describe().round(3))

    models = {
        "Close (no dividends)": smf.ols("exret_close ~ MKT_RF", data=data).fit(),
        "Adjusted close": smf.ols("exret_adj ~ MKT_RF", data=data).fit(),
    }
    for name, res in models.items():
        print(f"\nCAPM, {name}")
        print(coef_table(res))

    gap = models["Adjusted close"].params["Intercept"] - \
        models["Close (no dividends)"].params["Intercept"]
    print(f"\nYahoo's close price already accounts for stock splits but not "
          f"dividends, so returns\nbased on it miss the dividend income. Here "
          f"that understates alpha by {gap:.2f}% per month.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    x_grid = np.linspace(data["MKT_RF"].min(), data["MKT_RF"].max(), 200)
    for ax, (name, res), col in zip(axes, models.items(),
                                    ["exret_close", "exret_adj"]):
        b = res.params
        ax.scatter(data["MKT_RF"], data[col], s=12, alpha=0.6)
        ax.plot(x_grid, b["Intercept"] + b["MKT_RF"] * x_grid, color="red",
                linestyle="--", linewidth=2,
                label=f"alpha = {b['Intercept']:.2f}, beta = {b['MKT_RF']:.2f}")
        ax.set_title(f"AAPL - {name}")
        ax.set_xlabel("Market excess return (%)")
        ax.legend()
    axes[0].set_ylabel("AAPL excess return (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_capm_fit.png", dpi=150)
    plt.close(fig)


def factor_model_comparison(data):
    section("Part B: from CAPM to multi-factor models (adjusted returns)")
    source = data["factor_source"].iloc[0] if "factor_source" in data else "french"
    print(f"Factor source: {source}"
          + (" (ETF-based proxies, no CMA factor)" if source == "yahoo" else ""))

    # Add factors step by step; skip any factor the data does not contain
    labels = {"MKT_RF": "market", "SMB": "size", "HML": "value",
              "RMW": "profitability", "CMA": "investment", "MOM": "momentum"}
    steps = [["MKT_RF"], ["SMB", "HML"], ["RMW", "CMA"], ["MOM"]]
    formulas, factors = {}, []
    for new in steps:
        new = [f for f in new if f in data.columns]
        if not new:
            continue
        factors += new
        name = "CAPM" if not formulas else "+ " + ", ".join(labels[f] for f in new)
        formulas[name] = "exret_adj ~ " + " + ".join(factors)

    rows = []
    last = list(formulas)[-1]
    for name, formula in formulas.items():
        res = smf.ols(formula, data=data).fit()
        rows.append({
            "model": name,
            "alpha (%/month)": res.params["Intercept"],
            "alpha p-value": res.pvalues["Intercept"],
            "market beta": res.params["MKT_RF"],
            "R²": res.rsquared,
            "adj. R²": res.rsquared_adj,
        })
        if name == last:
            print(f"Full coefficient table, largest model ({formula}):")
            print(coef_table(res))

    comparison = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 120)
    print("\nModel comparison:")
    print(comparison.round(4))
    comparison.round(4).to_csv(FIG_DIR / "04_factor_model_comparison.csv")


if __name__ == "__main__":
    df = load_data()
    print(f"Sample: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m} "
          f"({len(df)} months)")
    capm_close_vs_adjusted(df)
    factor_model_comparison(df)
