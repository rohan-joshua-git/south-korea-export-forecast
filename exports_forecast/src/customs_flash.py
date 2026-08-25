"""Korea Customs Service first-10-day / first-20-day flash export figures.

This is arguably the single highest-value feature candidate in Layer 1 -
it's a partial read on the same month's exports, published before the
month ends, which is exactly the kind of lead indicator nowcasting papers
(Jaax et al., 2024) build around. It's also the one feature in this
project that turned out to have a real, binding data-quality cost, not
just a modeling one - see run_layer1_customs.py's findings before treating
this as a free win.

Source: data.go.kr's own API for this was blocked on manual signup (email
verification issue never resolved) and the originally-assumed dataset (id
15101636) was independently confirmed to be the wrong one anyway (monthly
port totals, not flash exports). Falls back instead to scraping Korea
Customs' public press releases directly from korea.kr's briefing room -
no login or key required, confirmed server-rendered and paginated, so the
full history can be enumerated without a browser. Each release ("N년 M월
D일 ~ M월 D일 수출입 현황 [잠정치]", title format has drifted several times
over the years - see TITLE_RE) attaches either a PDF or an HWP file
depending on the year; both are downloaded and parsed here (pypdf / pyhwp
respectively) for the same templated "총괄" summary line.

Real, reported coverage after building the full 2010-2026 scrape:
  - 131 releases discovered with a recognized title format (older releases
    use a structurally different title this scraper doesn't attempt to
    parse - a known gap, not a silent one).
  - 94 of those actually parsed to a figure. The rest failed for two
    distinct reasons, both left visible in build_flash_series()'s output
    rather than silently dropped: some HWP files' summary text lives
    inside table/text-box objects that pyhwp's plain-text extraction
    doesn't reach (mostly 2021-2022), and some older releases (2018-2020)
    didn't parse for a mix of the same reasons.
  - After merging with the rest of Layer 1's features and the lag terms,
    only 50 rows survive, spanning 2021-05 to 2026-04 - shorter than every
    other feature in this project. That's the real constraint: even with
    genuine, correctly-sourced data, the practically scrapeable history
    isn't long enough to run this feature through the assignment's
    36-month walk-forward minimum. See run_layer1_customs.py for how that
    gets handled (an explicitly-labeled exploratory check, not a Layer
    result on the same footing as Layers 0-2).
"""

import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_CACHE_DIR = DATA_DIR / "customs_pdfs"
LISTING_CACHE = DATA_DIR / "customs_flash_listing.csv"
PARSED_CACHE = DATA_DIR / "customs_flash.csv"

LIST_URL = "https://www.korea.kr/briefing/pressReleaseList.do"
DOWNLOAD_URL = "https://www.korea.kr/common/download.do"
HEADERS = {"User-Agent": "Mozilla/5.0"}
KCS_REP_CODE = "B00003"  # 관세청, confirmed from the site's own filter checkbox value

# Title format has drifted over the years - observed variants during a full
# 2015-2026 scrape:
#   "2026년 7월 1일 ~ 7월 20일 수출입 현황 [잠정치]"   (recent: 4-digit year, brackets)
#   "'25년 6월 1일 ~ 6월 20일 수출입 현황"              (2-digit year w/ apostrophe, no brackets)
#   "'21년 10월 1일 ∼ 10월 20일 수출입 현황"            (curly apostrophe, "∼" not "~")
#   "2018년 6월 1일 ∼ 6월 20일 수출입 현황"             (4-digit year, "∼")
# Older still ("5.1-20 수출입 현황", pre-~2018) uses a different structure
# entirely and is NOT matched here - a known, reported coverage gap rather
# than a silent one (see discover_releases' docstring/output).
TITLE_RE = re.compile(
    r"^['‘’]?(\d{2,4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*[~∼]\s*"
    r"(\d{1,2})월\s*(\d{1,2})일\s*수출입\s*현황(?:\s*[\[(]잠정치[\])])?$"
)


def _normalize_year(year_str):
    year = int(year_str)
    return 2000 + year if year < 100 else year

# The "총괄" (overview) line at the top of every release, in both the PDF
# and HWP versions, e.g.:
# "총괄 ㅇ (7.1.∼7.20.) 수출 549억 달러, 수입 427억 달러로 전년동기대비
#  수출 52.3%(188.7억 달러↑) 증가, 수입 20.0%(71.1억 달러↑) 증가"
# Chosen over the "<요약>" section's prose (which reads more naturally but
# turned out to be missing entirely from some HWP extractions) because this
# one was confirmed present in every sample checked, PDF and HWP alike.
#
# Matched against text with ALL whitespace stripped first (see
# _parse_figures) rather than against the raw extracted text: some years
# space out individual words ("전년 동기 대비") and others don't
# ("전년동기대비") with no consistent rule, and stripping whitespace
# entirely sidesteps that instead of enumerating every spacing variant.
FIGURE_RE = re.compile(
    r"수출([\d,]+(?:\.\d+)?)억달러,수입([\d,]+(?:\.\d+)?)억달러로전년동기대비"
    r"수출△?([\d.]+)%\([^)]*\)(증가|감소),수입△?([\d.]+)%\([^)]*\)(증가|감소)"
)


def _get(url, params=None, retries=3):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def _list_page(start_date, end_date, page_index):
    html = _get(LIST_URL, {
        "srchWord": "수출입 현황 잠정치",
        "startDate": start_date,
        "endDate": end_date,
        "repCode": KCS_REP_CODE,
        "pageIndex": page_index,
    }).decode("utf-8", errors="replace")
    return html


_ENTRY_RE = re.compile(
    r'newsId=(\d+)[^>]*>.*?<strong>(.*?)</strong>.*?<span>(\d{4}-\d{2}-\d{2})</span>',
    re.DOTALL,
)


def discover_releases(start_date="2015-01-01", end_date=None, max_pages=50):
    """Scrapes the korea.kr listing for 관세청's 10-day and 20-day flash
    export press releases. Returns a DataFrame: newsId, published, title,
    start/end of the covered period. Only titles matching the current
    "N년 M월 D일 ~ M월 D일 수출입 현황 [잠정치]" format are kept - excludes
    the whole-month "[확정치]"/"[잠정치]" releases (a different, coincident
    rather than leading, signal - see module docstring) and anything in an
    older/different title format this regex doesn't recognize."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    seen_ids = set()
    rows = []
    for page in range(1, max_pages + 1):
        html = _list_page(start_date, end_date, page)
        entries_this_page = 0
        for m in _ENTRY_RE.finditer(html):
            news_id, raw_title, published = m.group(1), m.group(2), m.group(3)
            entries_this_page += 1
            if news_id in seen_ids:
                continue
            seen_ids.add(news_id)
            title = re.sub(r"<[^>]+>", "", raw_title).strip()

            title_m = TITLE_RE.match(title)
            if not title_m:
                continue  # whole-month release, or an unrecognized older title format

            year_str, start_month, start_day, end_month, end_day = title_m.groups()
            year = _normalize_year(year_str)
            start_month, start_day, end_month, end_day = map(int, (start_month, start_day, end_month, end_day))
            period = "10d" if end_day <= 10 else "20d"
            rows.append({
                "news_id": news_id, "published": published, "title": title,
                "year": year, "start_month": start_month, "end_day": end_day,
                "period": period,
            })
        if entries_this_page == 0 and page > 1:
            break  # ran past the last page entirely (no results at all, matching or not)

    df = pd.DataFrame(rows).drop_duplicates("news_id").reset_index(drop=True)
    return df


def _find_attachment(news_id):
    """Returns (file_id, kind) for the release's attachment, preferring PDF
    but falling back to HWP - many releases from roughly 2025 and earlier
    only attach a .hwp, no PDF at all (confirmed by inspecting several
    detail pages directly), so PDF-only assumptions silently dropped most
    of the archive on the first pass."""
    html = _get(f"https://www.korea.kr/briefing/pressReleaseView.do?newsId={news_id}").decode(
        "utf-8", errors="replace"
    )
    pdf_m = re.search(r'/common/download\.do\?fileId=(\d+)&amp;tblKey=GMN"[^>]*>\s*<img[^>]*icon_ipdf', html)
    if pdf_m:
        return pdf_m.group(1), "pdf"
    hwp_m = re.search(r'/common/download\.do\?fileId=(\d+)&amp;tblKey=GMN"[^>]*>\s*<img[^>]*icon_ihangul', html)
    if hwp_m:
        return hwp_m.group(1), "hwp"
    return None, None


def _extract_hwp_text(hwp_path):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.txt"
        subprocess.run(
            ["hwp5txt", "--output", str(out_path), str(hwp_path)],
            check=True, capture_output=True,
        )
        return out_path.read_text(encoding="utf-8")


def _fetch_doc_text(news_id):
    PDF_CACHE_DIR.mkdir(exist_ok=True)
    text_cache = PDF_CACHE_DIR / f"{news_id}.txt"
    if text_cache.exists():
        return text_cache.read_text(encoding="utf-8")

    file_id, kind = _find_attachment(news_id)
    if file_id is None:
        return None

    doc_bytes = _get(DOWNLOAD_URL, {"fileId": file_id, "tblKey": "GMN"})
    doc_path = PDF_CACHE_DIR / f"{news_id}.{kind}"
    doc_path.write_bytes(doc_bytes)

    if kind == "pdf":
        reader = PdfReader(str(doc_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = _extract_hwp_text(doc_path)

    text_cache.write_text(text, encoding="utf-8")
    return text


def _parse_figures(text):
    m = FIGURE_RE.search(re.sub(r"\s+", "", text))
    if not m:
        return None
    export_100m_usd, import_100m_usd, export_pct, export_dir, import_pct, import_dir = m.groups()
    export_yoy = float(export_pct) * (1 if export_dir == "증가" else -1)
    import_yoy = float(import_pct) * (1 if import_dir == "증가" else -1)
    return {
        "export_usd": float(export_100m_usd.replace(",", "")) * 1e8,
        "export_yoy_pct": export_yoy,
        "import_usd": float(import_100m_usd.replace(",", "")) * 1e8,
        "import_yoy_pct": import_yoy,
    }


def build_flash_series(start_date="2015-01-01", verbose=True):
    """Scrapes, downloads, and parses every available 10-day/20-day flash
    release, caching each step (listing, PDFs, parsed text) so reruns are
    fast. Returns a DataFrame indexed by the covered period's month-start,
    one row per period ('10d' and '20d' as separate columns) - NOT
    reindexed to a full monthly series here, that alignment choice belongs
    to features.py, which decides how this gets used as a nowcast input
    for the target month."""
    if LISTING_CACHE.exists():
        listing = pd.read_csv(LISTING_CACHE, dtype={"news_id": str})
    else:
        listing = discover_releases(start_date=start_date)
        listing.to_csv(LISTING_CACHE, index=False)

    parsed_rows = []
    failures = []
    for _, row in listing.iterrows():
        text = _fetch_doc_text(row["news_id"])
        if text is None:
            failures.append((row["news_id"], row["title"], "no PDF or HWP attachment found"))
            continue
        figures = _parse_figures(text)
        if figures is None:
            failures.append((row["news_id"], row["title"], "regex did not match"))
            continue
        parsed_rows.append({
            "date": pd.Timestamp(year=row["year"], month=row["start_month"], day=1),
            "period": row["period"],
            "news_id": row["news_id"],
            **figures,
        })

    if verbose:
        print(f"discovered {len(listing)} releases, parsed {len(parsed_rows)}, "
              f"failed {len(failures)}")
        for news_id, title, reason in failures[:10]:
            print(f"  FAILED {news_id} ({reason}): {title}")

    df = pd.DataFrame(parsed_rows)
    if df.empty:
        return df
    df = df.sort_values(["date", "period"]).reset_index(drop=True)
    df.to_csv(PARSED_CACHE, index=False)
    return df


if __name__ == "__main__":
    df = build_flash_series()
    print(df.tail(10))
