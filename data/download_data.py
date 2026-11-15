"""
Download the data used in 04_asset_pricing.py and save it as
data/aapl_factors_monthly.csv.

Two sources are supported:

  --source yahoo  (default)
      Everything comes from Yahoo Finance via `yfinance`. The factors are
      built from ETF returns, a common practical proxy for the Fama-French
      factors:
        MKT_RF  S&P 500 ETF (SPY) return minus the risk-free rate
        SMB     small-cap (IWM)  minus large-cap (SPY)       -> size
        HML     value (IWD)      minus growth (IWF)          -> value
        RMW     quality (QUAL)   minus market (SPY)          -> profitability
        MOM     momentum (MTUM)  minus market (SPY)          -> momentum
        RF      13-week US T-bill yield (^IRX) / 12
      There is no clean ETF proxy for the investment factor (CMA).

  --source french
      Official Fama-French 5 factors + momentum from the Kenneth R. French
      Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
      data_library.html). The site is not reachable from every network; if a
      download fails, save the two zip files listed below in data/ manually.

All returns are monthly and in percent.

Usage:
    python data/download_data.py
    python data/download_data.py --start 2015-01 --end 2024-12
    python data/download_data.py --source french
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
OUT_FILE = DATA_DIR / "aapl_factors_monthly.csv"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}


# ----------------------------------------------------------------------------
# Yahoo Finance
# ----------------------------------------------------------------------------
def yahoo_monthly(tickers, start, end, field):
    """Monthly prices for a list of tickers, indexed by month."""
    prices = yf.download(tickers, start=f"{start}-01",
                         end=(pd.Period(end, "M") + 1).to_timestamp(),
                         interval="1mo", auto_adjust=False, progress=False)
    if prices.empty:
        sys.exit("Yahoo Finance returned no data. Check your internet "
                 "connection and try again.")
    table = prices[field]
    if isinstance(table, pd.Series):
        table = table.to_frame(tickers[0])
    table.index = pd.PeriodIndex(table.index, freq="M")
    table.index.name = "month"
    return table


def build_yahoo_factors(start, end):
    # Start one month early so the first month still has a return
    first = str(pd.Period(start, "M") - 1)

    etfs = ["SPY", "IWM", "IWD", "IWF", "QUAL", "MTUM"]
    ret = yahoo_monthly(etfs, first, end, "Adj Close").pct_change() * 100

    # ^IRX is an annualised yield in %; the rate known at the start of the
    # month (previous month-end) is used as that month's risk-free return.
    irx = yahoo_monthly(["^IRX"], first, end, "Close").iloc[:, 0]
    rf = (irx / 12).shift(1)

    factors = pd.DataFrame({
        "MKT_RF": ret["SPY"] - rf,
        "SMB": ret["IWM"] - ret["SPY"],
        "HML": ret["IWD"] - ret["IWF"],
        "RMW": ret["QUAL"] - ret["SPY"],
        "MOM": ret["MTUM"] - ret["SPY"],
        "RF": rf,
    })
    return factors.loc[start:end]


# ----------------------------------------------------------------------------
# Kenneth French Data Library
# ----------------------------------------------------------------------------
def fetch_french_zip(filename):
    """Use a manually saved zip in data/ if present, otherwise download it."""
    local = DATA_DIR / filename
    if local.exists():
        print(f"  using local file {local.name}")
        return local.read_bytes()
    try:
        from curl_cffi import requests as cffi_requests
        resp = cffi_requests.get(FRENCH_BASE + filename,
                                 impersonate="chrome", timeout=60)
        resp.raise_for_status()
    except Exception as err:
        sys.exit(f"\nCould not download {filename}: {err}\n"
                 f"Either use the default Yahoo source "
                 f"(python data/download_data.py),\nor download these files "
                 f"manually and save them (still zipped) in {DATA_DIR}:\n"
                 + "\n".join(f"  {FRENCH_BASE}{f}" for f in FRENCH_FILES.values()))
    local.write_bytes(resp.content)
    return resp.content


def read_french_monthly(filename):
    """Return the monthly table from a Kenneth French CSV zip."""
    with zipfile.ZipFile(io.BytesIO(fetch_french_zip(filename))) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    def is_month(line):
        first = line.split(",")[0].strip()
        return first.isdigit() and len(first) == 6

    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if is_month(line)) - 1
    header = ["yyyymm"] + [h.strip() for h in lines[header_idx].split(",")[1:]]

    rows = []
    for line in lines[header_idx + 1:]:
        if not is_month(line):
            break                      # monthly block ends; annual data follows
        rows.append([v.strip() for v in line.split(",")])

    table = pd.DataFrame(rows, columns=header)
    table["month"] = pd.PeriodIndex(table.pop("yyyymm"), freq="M")
    return table.set_index("month").astype(float)


def build_french_factors(start, end):
    ff5 = read_french_monthly(FRENCH_FILES["ff5"]).rename(columns={"Mkt-RF": "MKT_RF"})
    mom = read_french_monthly(FRENCH_FILES["mom"])
    mom.columns = ["MOM"]
    return ff5.join(mom, how="inner").loc[start:end]


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2023-12", help="last month, YYYY-MM")
    parser.add_argument("--source", choices=["yahoo", "french"], default="yahoo",
                        help="where the factors come from")
    args = parser.parse_args()

    print(f"Building factors from: {args.source}")
    if args.source == "yahoo":
        factors = build_yahoo_factors(args.start, args.end)
    else:
        factors = build_french_factors(args.start, args.end)

    print("Downloading AAPL prices ...")
    prices = yahoo_monthly(["AAPL"], args.start, args.end, ["Close", "Adj Close"])
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.rename(columns={"Close": "close", "Adj Close": "adj_close"})

    data = prices[["close", "adj_close"]].join(factors, how="inner").dropna()
    data.index = data.index.to_timestamp()
    data.index.name = "date"
    data["factor_source"] = args.source

    data.to_csv(OUT_FILE, float_format="%.6f")
    print(f"Saved {len(data)} months ({data.index.min():%Y-%m} to "
          f"{data.index.max():%Y-%m}) to {OUT_FILE}")


if __name__ == "__main__":
    main()
