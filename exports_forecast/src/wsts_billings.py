"""WSTS (World Semiconductor Trade Statistics) global semiconductor billings.

A direct read on the memory/semiconductor cycle - actual industry sales
revenue by region, not a market-price proxy the way the existing
sox/samsung/skhynix features are (those capture what investors expect the
cycle to do; this captures what it's actually doing). Free, no login,
monthly data back to 1986: https://www.wsts.org/67/Historical-Billings-Report
("36 Years WSTS Blue Book Data", confirmed by inspecting the downloaded
file directly - a static xlsx URL that includes the report month, so this
needs occasional re-pointing as WSTS publishes new months).

Asia Pacific is used rather than Worldwide: Korea's memory exports are far
more exposed to the Asia Pacific semiconductor supply chain (fabs,
assembly/test) than to Americas/Europe/Japan billings, which mix in demand
segments (auto, industrial) Korea's DRAM/NAND-heavy export base is less
tied to.
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "wsts_billings.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Static per-month URL from wsts.org - if this 404s, check
# https://www.wsts.org/67/Historical-Billings-Report for the current link
# and update this constant (the report is republished monthly with the
# month in the filename).
BLUE_BOOK_URL = (
    "https://www.wsts.org/esraCMS/extension/media/f/WST/7676/"
    "WSTS-Historical-Billings-Report-May_2026.xlsx"
)

MONTH_COLS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _download():
    req = urllib.request.Request(BLUE_BOOK_URL, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=60).read()


def build_billings_series(region="Asia Pacific"):
    """Returns a monthly pd.Series of WSTS billings (thousand USD) for the
    given region, indexed on month-start. Cached to CACHE after first
    parse - re-download the xlsx (update BLUE_BOOK_URL first) and delete
    the cache to refresh with newer months."""
    if CACHE.exists():
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        return df[region].dropna().rename(f"wsts_{region.lower().replace(' ', '_')}")

    xlsx_path = DATA_DIR / "wsts_blue_book.xlsx"
    if not xlsx_path.exists():
        xlsx_path.write_bytes(_download())

    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Monthly Data"]

    rows = {}
    current_year = None
    for row in ws.iter_rows(values_only=True):
        label = row[0]
        if isinstance(label, int):
            current_year = label
            continue
        if label in ("Americas", "Europe", "Japan", "Asia Pacific", "Worldwide") and current_year:
            for month_idx, value in enumerate(row[1:13], start=1):
                if value is None:
                    continue
                date = pd.Timestamp(year=current_year, month=month_idx, day=1)
                rows.setdefault(label, {})[date] = value

    df = pd.DataFrame(rows).sort_index()
    df.index.name = "date"
    df.to_csv(CACHE)
    return df[region].dropna().rename(f"wsts_{region.lower().replace(' ', '_')}")


if __name__ == "__main__":
    s = build_billings_series()
    print(s.tail(12))
    print(f"\n{len(s)} monthly rows, {s.index.min().date()} to {s.index.max().date()}")
