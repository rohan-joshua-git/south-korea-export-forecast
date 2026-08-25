"""Generates 05_customs_flash_exploratory.ipynb. Run once; the notebook
itself is the deliverable, this script is just how it's assembled."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Customs Flash Export Feature: Exploratory, Not a Layer

## Introduction

Layers 1 and 2 both flagged the same open item: Korea Customs' first-10-day
and first-20-day flash export figures are the one feature in this project
that's a genuine *leading* indicator rather than a coincident one (unlike
FX, oil, SOX, or even Samsung's own stock price, all of which move
alongside exports rather than ahead of them). This notebook chases that
down for real and reports what actually happened - both the sourcing
problem and the data-depth problem that followed it.

**This is explicitly not "Layer 3."** It doesn't meet the assignment's
36-month walk-forward minimum, for reasons explained below. It's reported
on its own, separate from the headline numbers in notebooks 01-04, with
the same discipline notebook 06's capstone applies to its own best-cell
finding: a real, honestly-reported result, clearly labeled as short of
the bar needed to call it validated.

---
## Methodology: sourcing

The original plan (data.go.kr's Korea Customs API) hit two problems.
First, account signup got stuck on an email verification step that never
resolved. Second, and independently worse: the dataset originally assumed
to be the flash-export series (id `15101636`) turned out, on checking the
page directly, to be monthly port-level trade totals - not preliminary
10-day figures at all. Registering a key for it would not have produced
the intended feature even if signup had worked.

The actual data source used here instead: Korea Customs Service's own
public press releases, hosted on korea.kr's briefing room
(`pressReleaseList.do`), no login required. Releases titled e.g. "2026년
7월 1일 ~ 7월 20일 수출입 현황 [잠정치]" are published on a fixed 10-day
cadence (1-10 day figures ~day 11, 1-20 day figures ~day 21) with a PDF or
HWP attachment containing a templated summary line:

> 수출 549억 달러, 수입 427억 달러로 전년동기대비 수출 52.3%(...) 증가,
> 수입 20.0%(...) 증가

`src/customs_flash.py` scrapes the listing (paginated, server-rendered
HTML), downloads each attachment, and regex-parses that line - handling
both PDF (`pypdf`) and HWP (`pyhwp`) since which format is attached
varies by year, and several real phrasing/formatting variants found
along the way (repeated vs. single "전년동기대비", a "△" decrease marker,
inconsistent word-spacing across years, and at least four different
title formats for the year field alone).

## Methodology: coverage

Of 131 releases found with a recognized title format, 94 parsed to a
clean figure. The rest failed for two distinct, logged reasons rather
than silently disappearing: some 2021-2022 HWP files store their summary
text inside table/text-box objects that pyhwp's plain-text extraction
doesn't reach, and some pre-2021 releases use an older title format this
scraper doesn't attempt to parse at all (a known, reported gap).

The practical constraint is what happens after merging with the rest of
Layer 1's features and the `target_lag1`/`target_lag12` terms: only 50
rows survive, spanning 2021-05 to 2026-04. That's shorter than every
other feature in this project, BDRY's 2018-03 start included."""))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, "../src")

from layer1_model import build_dataset, make_gbm_forecaster, make_ridge_forecaster
from walk_forward import walk_forward_x, rmse, mape

y_with, X_with = build_dataset(include_customs_flash=True)
y_without, X_without = build_dataset(include_customs_flash=False)

# align the "without" dataset to the exact same dates as the "with" one,
# so this is an apples-to-apples comparison, not confounded by window choice
common_index = y_with.index
y_without, X_without = y_without.loc[common_index], X_without.loc[common_index]

print(f"WITH customs_flash:    {len(y_with)} rows, {X_with.shape[1]} features, "
      f"{y_with.index.min().date()} to {y_with.index.max().date()}")
print(f"WITHOUT (same window): {len(y_without)} rows, {X_without.shape[1]} features")"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Methodology: the exploratory backtest

`walk_forward_x`'s default `min_train=36` combined with 50 total rows caps
the test window at 14 origins - far under the assignment's 36-month
minimum. Rather than quietly shrinking the test window and reporting the
result as if it were on the same footing as Layers 0-2, it's run here
explicitly labeled as exploratory, alongside a same-window,
same-everything-else comparison *without* the customs feature - the only
way to isolate whether the feature itself moves anything, since a shorter,
more recent window is a different (and on its own, not obviously harder or
easier) problem than Layer 1's original 98-row window."""))

cells.append(nbf.v4.new_code_cell("""\
N_TEST = 14   # NOT the assignment's 36-month minimum - see markdown above
MIN_TRAIN = 30

variants = {
    "GBM, with customs_flash": (X_with, make_gbm_forecaster()),
    "Ridge, with customs_flash": (X_with, make_ridge_forecaster(alpha=10.0)),
    "GBM, without (same window)": (X_without, make_gbm_forecaster()),
    "Ridge, without (same window)": (X_without, make_ridge_forecaster(alpha=10.0)),
}

for name, (X, fn) in variants.items():
    results = walk_forward_x(y_with, X, fn, n_test=N_TEST, min_train=MIN_TRAIN)
    print(f"{name:>32}  RMSE={rmse(results):>14,.0f}  MAPE={mape(results):>6.2f}%")"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Findings

Both models improve slightly with the customs flash feature added, on
this one small window: Ridge MAPE goes from 3.62% to 3.56%, GBM from
6.11% to 5.97%. Small, consistent in direction across both models (the
same cross-check Layer 2's sentiment feature was judged by), and notably
this is the *only* alt-data feature tried anywhere in this project that
improved GBM as well as Ridge - everywhere else, GBM only ever got worse
when a feature was added, consistent with it being past its complexity
budget for the sample sizes available. That's a mild point in favor of
this feature carrying real signal rather than noise.

It is also, on 14 origins, not a claim this project is prepared to stand
behind the way it stands behind the Layer 0-2 numbers. Both the absolute
MAPE levels here (5.97-6.11% for GBM) and the comparison itself are noisy
at this sample size in a way 36 origins would meaningfully reduce, and the
2021-2026 window is not the same test Layers 0-2 were judged against, so
none of these numbers are comparable to the notebook 00 summary table.

## Key takeaways

The sourcing problem (data.go.kr blocked, wrong dataset originally
assumed) was solvable without a working API key - the press-release
scrape is real, verified, first-party government data, not a proxy or a
workaround.

The remaining problem is depth, not access. Practically scrapeable
history caps out around 2021-05, which is the actual reason this can't be
reported as a validated Layer result yet - not a modeling limitation, a
historical-data-availability one. Getting a genuinely valid 36-month
walk-forward on this feature is now mostly a waiting problem (more months
accumulate naturally) or a second-source problem (finding pre-2021 flash
figures somewhere with a more machine-readable archive), not a scraping
or parsing one.

Given the direction of the result and that this is the only feature in
the whole project to help both models rather than just the regularized
one, this is the most promising remaining lead if the project continues -
more promising than it looked when it was simply "blocked on an API key,"
and a genuinely different kind of constraint than the sample-size story
that explained every other layer's null result.

Next: notebook 06 steps back from individual features entirely and asks
the project's central question directly - does more data or more
complexity help, exhaustively, on one fixed dataset - closing the loop
this notebook and notebook 04 both left open."""))

nb["cells"] = cells
nbf.write(nb, "05_customs_flash_exploratory.ipynb")
print("wrote 05_customs_flash_exploratory.ipynb")
