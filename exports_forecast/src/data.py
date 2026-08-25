"""Loads the target series: South Korea monthly total exports (USD, SA).

Source: FRED series XTEXVA01KRM667S (OECD MEI, merchandise trade, value,
exchange-rate converted, seasonally adjusted). Downloaded straight from
FRED's CSV endpoint so there's no API key to manage.
"""

import pandas as pd
from pathlib import Path

FRED_SERIES_ID = "XTEXVA01KRM667S"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "korea_exports_raw.csv"


def download_raw(force=False):
    if RAW_PATH.exists() and not force:
        return pd.read_csv(RAW_PATH, index_col=0, parse_dates=True).iloc[:, 0]

    df = pd.read_csv(FRED_CSV_URL)
    df.columns = ["date", "exports"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH)
    return df["exports"]


def load_exports(start="1990-01-01"):
    series = download_raw()
    series = series[series.index >= start]
    series.name = "exports_usd"
    return series


if __name__ == "__main__":
    s = load_exports()
    print(s.tail())
    print(f"\n{len(s)} monthly observations, {s.index.min().date()} to {s.index.max().date()}")
