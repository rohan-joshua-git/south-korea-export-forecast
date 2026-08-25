"""IMF PortWatch: satellite/AIS-tracked daily port activity and trade
volume estimates for Busan, South Korea's largest container port and the
dominant gateway for its export trade.

Free, no login, no API key - confirmed by querying the live ArcGIS
FeatureServer directly. This is the validated, IMF-published version of
the kind of idea this project's shelved "Layer 3: VIIRS nightlight around
Busan port" was reaching for (see README) - satellite-based real-time
trade tracking is an active nowcasting research area (Arslanalp et al.,
"Nowcasting Global Trade from Space", IMF WP/25/93, 2025), and PortWatch is
the IMF's own production version of it, not a DIY proxy.

Data: daily port calls, and import/export trade volume estimates in metric
tons, derived from AIS vessel-tracking signals, for 2065 ports worldwide.
Busan is portid='port1065' (confirmed live). History starts 2019-01-01
(confirmed via returnCountOnly query: 2769 daily rows through 2026-07-31,
consistent with a 2019-01-01 start and no gaps) - this is tonnage moving
through the port, a physical-volume proxy for trade activity, not a
dollar-denominated export figure, so it's used here as an exogenous
feature, not a replacement for the target series.
"""

import time
import urllib.error
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "portwatch_busan.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

QUERY_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Ports_Data/FeatureServer/0/query"
BUSAN_PORT_ID = "port1065"
PAGE_SIZE = 2000  # ArcGIS FeatureServer's transfer limit for this service


def _query(offset):
    params = {
        "where": f"portid='{BUSAN_PORT_ID}'",
        "outFields": "date,portcalls,import,export",
        "orderByFields": "date ASC",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "f": "json",
    }
    url = f"{QUERY_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(2)


def _fetch_all_daily():
    # Don't infer "last page" from a short response - the service silently
    # caps each response below PAGE_SIZE for this dataset regardless of
    # resultRecordCount (confirmed live: requesting 2000 only ever returns
    # ~1000), which made an earlier version of this loop stop after page 1.
    # Only an empty response reliably means "no more data".
    rows, offset = [], 0
    while True:
        payload = _query(offset)
        features = payload.get("features", [])
        if not features:
            break
        rows.extend(f["attributes"] for f in features)
        offset += len(features)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def build_busan_export_series():
    """Monthly Busan export tonnage (sum of daily estimates), indexed on
    month-start. Cached after first fetch - delete CACHE to refresh with
    newer daily rows."""
    if CACHE.exists():
        daily = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    else:
        daily = _fetch_all_daily()
        daily.to_csv(CACHE)

    monthly = daily["export"].resample("MS").sum()
    return monthly.rename("busan_export_tons")


if __name__ == "__main__":
    s = build_busan_export_series()
    print(s.tail(12))
    print(f"\n{len(s)} monthly rows, {s.index.min().date()} to {s.index.max().date()}")
