"""Layer 1 feature loaders: structured alt-data, no API keys required.

Each loader returns a monthly pd.Series, indexed on month-start to match
data.load_exports(). Sources and what they stand in for:

  usd_krw   - FRED DEXKOUS, daily          -> won weakness/strength
  oil       - FRED DCOILWTICO, daily       -> input costs, freight, demand proxy
  sox       - Yahoo ^SOX, monthly          -> global semiconductor cycle
  samsung   - Yahoo 005930.KS, monthly     -> memory-chip demand proxy
  skhynix   - Yahoo 000660.KS, monthly     -> memory-chip demand proxy
  bdry      - Yahoo BDRY, monthly          -> Baltic Dry Index proxy (only
                                               free option found; the actual
                                               BDI is not published for free
                                               anywhere we could locate)
  trends    - Google Trends via pytrends   -> forward search interest

  customs_flash_yoy - Korea Customs press releases, 10-day/20-day
                                            -> flash export growth, the
                                               only genuinely leading (not
                                               coincident) feature here;
                                               see customs_flash.py for
                                               sourcing and its real
                                               constraint: parseable
                                               coverage only reaches back
                                               to 2021-05, shorter than
                                               every other feature, so
                                               including it shrinks the
                                               usable sample further

  wsts_asia_pacific - WSTS Blue Book, monthly -> actual semiconductor
                                               billings revenue (Asia
                                               Pacific region), a direct
                                               read on the memory cycle
                                               rather than a market-price
                                               proxy; see wsts_billings.py
  busan_export_tons - IMF PortWatch, daily    -> satellite/AIS-tracked
                                               physical export tonnage
                                               through Busan; see
                                               portwatch.py. Starts
                                               2019-01, the binding
                                               constraint on any dataset
                                               that includes it.

Known gaps, not faked:
  - DRAM/NAND spot prices (DRAMeXchange/TrendForce): paywalled. Samsung and
    SK Hynix equity prices are used as a market-based proxy instead - the
    market prices in memory-cycle expectations continuously, which is
    arguably a more forward-looking signal than a lagging spot price anyway,
    but it's a substitution, not the real thing, and should be named as such.
"""

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=max&interval=1mo"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_fred_monthly(series_id, agg="mean"):
    cache = DATA_DIR / f"{series_id}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        df = pd.read_csv(FRED_URL.format(id=series_id))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df["value"].dropna()
        s.to_csv(cache)
    return s.resample("MS").agg(agg)


def _fetch_yahoo_monthly(ticker, cache_name):
    cache = DATA_DIR / f"{cache_name}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s

    req = urllib.request.Request(YAHOO_URL.format(ticker=ticker), headers=HEADERS)
    payload = json.loads(urllib.request.urlopen(req).read())
    result = payload["chart"]["result"][0]

    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(timestamps, unit="s").tz_localize(None)
    s = pd.Series(closes, index=idx).dropna()
    s.index = s.index.to_period("M").to_timestamp()
    s = s.groupby(s.index).mean()
    s.to_csv(cache)
    return s


def usd_krw():
    return _fetch_fred_monthly("DEXKOUS").rename("usd_krw")


def oil():
    return _fetch_fred_monthly("DCOILWTICO").rename("oil_wti")


def sox():
    return _fetch_yahoo_monthly("%5ESOX", "sox").rename("sox")


def samsung():
    return _fetch_yahoo_monthly("005930.KS", "samsung").rename("samsung")


def skhynix():
    return _fetch_yahoo_monthly("000660.KS", "skhynix").rename("skhynix")


def bdry():
    return _fetch_yahoo_monthly("BDRY", "bdry").rename("bdry")


def customs_flash_yoy():
    """Monthly series of the flash export YoY growth rate (%), from Korea
    Customs' 10-day/20-day press releases (see customs_flash.py). Prefers
    the 1-20-day reading (more of the month observed) and falls back to
    the 1-10-day reading for months where only that one parsed - both are
    "observable by end of month" under this project's existing nowcast
    convention (same standard as the full-month averages used for
    usd_krw/oil/sox elsewhere in this file), and the 20-day figure is in
    fact available earlier in the month than a full-month average would
    be.

    Real constraint, not a rounding error: parseable coverage only starts
    2021-05 (see customs_flash.py's build script output for exactly why -
    older releases are scanned PDF/HWP image-style documents this
    project's parser can't reliably extract from). That's a shorter
    window than every other feature here, so including this one shrinks
    whatever dataset it's added to down to that overlap - a real cost
    against its status as "the most promising lead", not a free add."""
    cache = DATA_DIR / "customs_flash.csv"
    if not cache.exists():
        from customs_flash import build_flash_series
        build_flash_series()

    df = pd.read_csv(cache, parse_dates=["date"])
    pivot = df.pivot_table(index="date", columns="period", values="export_yoy_pct")
    combined = pivot.get("20d")
    if combined is None:
        combined = pivot.get("10d")
    else:
        combined = combined.fillna(pivot.get("10d"))
    return combined.rename("customs_flash_yoy")


def wsts_asia_pacific():
    from wsts_billings import build_billings_series
    return build_billings_series(region="Asia Pacific")


def busan_export_tons():
    from portwatch import build_busan_export_series
    return build_busan_export_series()


def google_trends(keyword="semiconductor exports", geo="KR"):
    cache = DATA_DIR / "trends.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s

    from pytrends.request import TrendReq

    pt = TrendReq(hl="en-US", tz=540)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    pt.build_payload([keyword], geo=geo, timeframe=f"2015-01-01 {end}")
    weekly = pt.interest_over_time()[keyword]
    s = weekly.resample("MS").mean().rename("trends")
    s.to_csv(cache)
    return s


def build_feature_matrix(include_customs_flash=False):
    """include_customs_flash defaults to False: customs_flash_yoy only
    covers 2021-05 onward (see its docstring), so turning it on shrinks
    the merged dataset's date range to match - a real tradeoff, not a
    free addition, so it's opt-in rather than silently changing every
    existing Layer 1/2 result. See run_layer1_customs.py for the isolated
    test of whether that tradeoff is worth it."""
    loaders = [usd_krw, oil, sox, samsung, skhynix, bdry, google_trends]
    if include_customs_flash:
        loaders = loaders + [customs_flash_yoy]
    frame = pd.concat([fn() for fn in loaders], axis=1)
    frame.index.name = "date"
    return frame


if __name__ == "__main__":
    X = build_feature_matrix()
    print(X.tail())
    print(f"\n{X.shape[0]} rows, columns: {list(X.columns)}")
    print("\nfirst valid index per column:")
    print(X.apply(lambda c: c.first_valid_index()))
