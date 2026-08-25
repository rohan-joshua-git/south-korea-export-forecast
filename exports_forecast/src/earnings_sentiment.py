"""Layer 2: guidance sentiment from Samsung Electronics earnings conference
call decks.

This is real, first-party public data - Samsung posts English-language
conference call PDFs on its IR site (samsung.com/global/ir) covering
management's demand/capex/outlook commentary each quarter, no login or key
needed. Scored with the Loughran-McDonald financial sentiment dictionary
(via pysentiment2) rather than a generic sentiment model, since LM was
built specifically for the kind of hedged, domain-specific language that
shows up in earnings materials (a generic model reads "impairment" or
"headwind" as neutral; LM knows they're not).

SK Hynix and Hyundai are not included in this pass - their IR sites don't
expose predictable, scrapeable PDF links the way Samsung's does, and DART
(dart.fss.or.kr) needs a manually-registered OpenDART API key, same
constraint as the Customs flash data in Layer 1. Samsung alone is a
reasonable proxy given its share of Korea's semiconductor exports, but
it's a scope limitation worth being upfront about, not a finished feature.

Leakage handling: a quarter's call is treated as "known" starting the month
*after* Samsung's approximate release month, which is deliberately
conservative (release is usually late in the release month itself, so this
adds a buffer instead of assuming same-month availability).
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
import pysentiment2 as ps

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_CACHE_DIR = DATA_DIR / "samsung_pdfs"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Scraped directly from https://www.samsung.com/global/ir/financial-information/earnings-release/
# (quarter, url) - naming is inconsistent quarter to quarter (path prefix,
# occasional "_01" suffix), so this is the verified list, not a guessed pattern.
SAMSUNG_CALL_URLS = {
    (2018, 1): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2018_1Q_conference_eng.pdf",
    (2018, 2): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2018_2Q_conference_eng.pdf",
    (2018, 3): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2018_3Q_conference_eng.pdf",
    (2018, 4): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2018_4Q_conference_eng.pdf",
    (2019, 1): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2019_1Q_conference_eng.pdf",
    (2019, 2): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2019_2Q_conference_eng.pdf",
    (2019, 3): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2019_3Q_conference_eng.pdf",
    (2019, 4): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2019_4Q_conference_eng.pdf",
    (2020, 1): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2020_1Q_conference_eng.pdf",
    (2020, 2): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2020_2Q_conference_eng.pdf",
    (2020, 3): "https://images.samsung.com/is/content/samsung/p5/global/ir/docs/2020_3Q_conference_eng.pdf",
    (2020, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2020_4Q_conference_eng.pdf",
    (2021, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2021_1Q_conference_eng.pdf",
    (2021, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2021_2Q_conference_eng.pdf",
    (2021, 3): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2021_3Q_conference_eng.pdf",
    (2021, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2021_4Q_conference_eng.pdf",
    (2022, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2022_1Q_conference_eng.pdf",
    (2022, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2022_2Q_conference_eng.pdf",
    (2022, 3): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2022_3Q_conference_eng.pdf",
    (2022, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2022_4Q_conference_eng.pdf",
    (2023, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2023_1Q_conference_eng.pdf",
    (2023, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2023_2Q_conference_eng.pdf",
    (2023, 3): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2023_3Q_conference_eng_01.pdf",
    (2023, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2023_4Q_conference_eng_01.pdf",
    (2024, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_1Q_conference_eng.pdf",
    (2024, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_2Q_conference_eng.pdf",
    (2024, 3): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_3Q_conference_eng.pdf",
    (2024, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2024_4Q_conference_eng.pdf",
    (2025, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2025_1Q_conference_eng.pdf",
    (2025, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2025_2Q_conference_eng.pdf",
    (2025, 3): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2025_3Q_conference_eng.pdf",
    (2025, 4): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2025_4Q_conference_eng.pdf",
    (2026, 1): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2026_1Q_conference_eng.pdf",
    (2026, 2): "https://images.samsung.com/is/content/samsung/assets/global/ir/docs/2026_2Q_conference_eng.pdf",
}

# Approximate release month for each quarter's call (Samsung's actual
# practice: Q1 late Apr, Q2 late Jul, Q3 late Oct, Q4 preliminary late Jan
# of the following year).
RELEASE_MONTH = {1: 4, 2: 7, 3: 10, 4: 1}


def _fetch_pdf_text(year, quarter):
    PDF_CACHE_DIR.mkdir(exist_ok=True)
    cache = PDF_CACHE_DIR / f"{year}Q{quarter}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    url = SAMSUNG_CALL_URLS[(year, quarter)]
    pdf_path = PDF_CACHE_DIR / f"{year}Q{quarter}.pdf"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        pdf_path.write_bytes(resp.read())

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    cache.write_text(text, encoding="utf-8")
    return text


def _release_date(year, quarter):
    month = RELEASE_MONTH[quarter]
    release_year = year + 1 if quarter == 4 else year
    return pd.Timestamp(year=release_year, month=month, day=1)


def build_sentiment_series():
    lm = ps.LM()
    rows = []
    for (year, quarter) in sorted(SAMSUNG_CALL_URLS):
        text = _fetch_pdf_text(year, quarter)
        tokens = lm.tokenize(text)
        score = lm.get_score(tokens)

        # available starting the month *after* the approximate release
        # month - conservative buffer against look-ahead bias
        available_from = _release_date(year, quarter) + pd.DateOffset(months=1)

        rows.append({
            "available_from": available_from,
            "polarity": score["Polarity"],
            "positive": score["Positive"],
            "negative": score["Negative"],
        })

    df = pd.DataFrame(rows).set_index("available_from").sort_index()

    monthly_index = pd.date_range(df.index.min(), pd.Timestamp.today(), freq="MS")
    monthly = df.reindex(monthly_index, method="ffill")
    monthly.index.name = "date"
    return monthly["polarity"].rename("samsung_sentiment")


if __name__ == "__main__":
    s = build_sentiment_series()
    print(s.tail(12))
    print(f"\n{len(s)} monthly rows, {s.index.min().date()} to {s.index.max().date()}")
