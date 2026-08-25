"""Generates 00_summary.ipynb. Run once; the notebook itself is the
deliverable, this script is just how it's assembled.

Reads the backtest CSVs every other notebook already wrote, rather than
hand-typing numbers, so this table can't drift from the per-layer results."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Summary: South Korea Monthly Total Exports Forecast

Q1 (Macro QR), NUS Investment Society recruitment case. **Start here** -
this notebook is a 60-second summary of notebooks 01-06. Full methodology,
limitations, and interpretability results are in the per-layer notebooks
this links to at the bottom.

## TL;DR

A random-walk baseline is genuinely hard to beat on this series at one
month ahead. Layers 1-2 (structured alt-data, then earnings-call
sentiment, routed through GBM/Ridge on top of lag features) were
benchmarked out-of-sample against that baseline and against each other,
and neither beat it on a like-for-like 36-month walk-forward test. A
different approach did move the needle: SARIMAX - SARIMA's proven
structure, given real exogenous data (global semiconductor billings)
directly rather than through a separate regressor - gets the best point
estimate in the project (2.50% MAPE vs. SARIMA's 2.92%), beating even the
SARIMA+Ridge blend. **Still not statistically confirmed** (Diebold-Mariano
p=0.17) at this sample size, so it's reported the same way the blend was -
a genuinely promising point estimate, not a proven result. Added
complexity keeps having to earn its keep out of sample; this is the first
time an addition earned as much of it as the blend did, and it did so with
real new data rather than a recombination of existing models."""))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, "../src")

import pandas as pd
from walk_forward import rmse, mape
from diebold_mariano import diebold_mariano

BACKTESTS = {
    "Naive (random walk)":              ("../data/backtest_naive.csv",           "Layer 0"),
    "Seasonal naive":                   ("../data/backtest_seasonal_naive.csv",  "Layer 0"),
    "SARIMA":                           ("../data/backtest_sarima.csv",          "Layer 0 - baseline"),
    "GBM + alt-data":                   ("../data/backtest_gbm_layer1.csv",      "Layer 1"),
    "Ridge + alt-data":                 ("../data/backtest_ridge_layer1.csv",    "Layer 1"),
    "GBM + alt-data + sentiment":       ("../data/backtest_gbm_layer2.csv",      "Layer 2"),
    "Ridge + alt-data + sentiment":     ("../data/backtest_ridge_layer2.csv",    "Layer 2"),
    "SARIMA + Ridge blend":             ("../data/backtest_blend.csv",           "Ensemble"),
    "SARIMAX + WSTS billings":          ("../data/backtest_sarimax_wsts_full.csv", "SARIMAX"),
}

sarima_df = pd.read_csv(BACKTESTS["SARIMA"][0], index_col=0, parse_dates=True)

rows = []
for name, (path, layer) in BACKTESTS.items():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    r, m = rmse(df), mape(df)

    if name == "SARIMA":
        sig = "- (baseline)"
    elif m < mape(sarima_df):
        aligned_sarima, aligned_df = sarima_df.align(df, join="inner")
        stat, p = diebold_mariano(
            aligned_sarima["actual"].values,
            aligned_sarima["forecast"].values,
            aligned_df["forecast"].values,
        )
        sig = f"Not significant (DM p={p:.2f})" if p >= 0.05 else f"Significant (DM p={p:.2f})"
    else:
        sig = "No - worse than baseline"

    rows.append({
        "Model": name, "Layer": layer, "n (OOS months)": len(df),
        "MAPE": f"{m:.2f}%", "RMSE (USD)": f"{r:,.0f}",
        "Beats SARIMA baseline?": sig,
    })

summary = pd.DataFrame(rows).set_index("Model")
summary"""))

cells.append(nbf.v4.new_markdown_cell("""\
A separate capstone experiment (notebook 06) scanned an 80-cell grid of
feature count x Ridge regularization strength and found one cell (10
features, alpha=100) with RMSE 2.41B, below every row above except
SARIMAX+WSTS. It is **deliberately excluded from this table**: it was
found by scanning 80 test-window evaluations rather than chosen on a
held-out validation window the way SARIMA's order and the blend's weight
were, so treating it as a result would repeat the exact test-set-tuning
mistake this project otherwise avoids. See notebook 06 for why the shape
of that surface, not its single best cell, is the trustworthy finding -
and for how it closes the loop on this whole project's Occam's razor
question.

## Key takeaways

1. **The baseline is a genuinely hard bar, not a strawman.** SARIMA
   (2.92% MAPE) barely separates from a plain random walk (3.14%). That
   closeness is itself informative: at one month ahead, most of the
   exploitable signal in this series is already in its own recent
   history.
2. **Neither structured alt-data nor earnings-call sentiment beat that
   bar, and the reason is diagnosed, not just observed.** SHAP
   (notebooks 03-04) shows the autoregressive term dominates; the
   GBM-vs-Ridge gap at ~90-98 training rows shows flexible models overfit
   before regularized ones do; the capstone surface (06) then confirms
   directly that more features need more regularization, and without it,
   more features actively hurt.
3. **Two independent point estimates now beat SARIMA, and both are
   honestly reported as unproven, not adopted.** The SARIMA + Ridge blend
   (2.65% MAPE) had its weight chosen on a validation window, never the
   test set, and still fails a Diebold-Mariano test at n=36 (p=0.15).
   SARIMAX with real semiconductor billings data (2.50% MAPE, notebook 02)
   does better still, on the full 436-row training history, and also
   isn't significant (p=0.17). Two structurally different approaches
   landing in the same place - better point estimate, not confirmable at
   this sample size - is itself informative, not just a repeated null
   result.
4. **The single most promising lead was chased down, and the honest
   answer is "not yet provable, for a data reason rather than a modeling
   one."** Korea Customs' flash export figures are a genuine leading
   indicator (unlike everything used so far, which is coincident at
   best). The data.go.kr API path was a dead end (blocked signup, and the
   originally-targeted dataset turned out to be the wrong one anyway), so
   this was rebuilt from Korea Customs' public press releases directly -
   real data, not a proxy. It parses cleanly for 94 of 131 releases, but
   the practically scrapeable history only reaches back to 2021-05,
   leaving 50 rows once merged - short of the assignment's 36-month
   minimum. A clearly-labeled exploratory check on the 14 origins that
   *are* available shows both Ridge and GBM improving slightly with the
   feature added, which is directionally the most encouraging single
   result in this project - but it is reported as exactly that,
   exploratory, not folded into Layer 1's headline numbers. See
   `src/customs_flash.py` and `src/run_layer1_customs.py`.
5. **Fitting real exogenous data directly into SARIMA's structure beat
   routing it through a separate GBM/Ridge on lag features (Layers 1-2's
   approach).** Global semiconductor billings (WSTS) and satellite-tracked
   Busan port trade volume (IMF PortWatch) are both real, freely-sourced,
   no-key-required data - researched from the trade-nowcasting literature
   rather than reused from what was already on hand. Busan's short
   history (2019+) reproduces the same sample-size problem the Customs
   feature hit; WSTS's much longer history (1986+) let the comparison run
   at full sample length, and that's the version that actually beat
   SARIMA. See notebook 02.

## Why this matters for a fund, not just an econometrics exercise

Korea's exports are a monthly, semiconductor-heavy proxy for global tech
demand, and the official print lags the reference month by roughly
five to six weeks. A forecast that reliably nowcasts that print ahead of
release would be a positioning input for KRW and semiconductor
supply-chain-exposed equities (Samsung, SK Hynix, and their downstream
peers) around the release date - trade the nowcast, not the stale
consensus. **That is not yet what this project has**: no layer here has
a statistically validated edge over SARIMA, so nothing above should be
read as a deployable signal today. What this project does establish is
the discipline and the harness (honest walk-forward validation,
validation/test separation, significance testing) that a real leading
indicator - most plausibly the Customs flash feature - would need to
be run through before anyone trades on it.

## What's next

- **Confirm the SARIMAX+WSTS result on a genuinely fresh window.** The
  most defensible next step given two unconfirmed point estimates now
  beating SARIMA (the blend and SARIMAX+WSTS) is not to keep re-testing
  against the same 36 months, but to pre-register a confirmation test on
  new data as it arrives, or extend the test window once more history
  exists.
- **Once Busan PortWatch accumulates more history (2019+, currently ~7
  years), retry SARIMAX with it included** - Test 1 in notebook 02 shows
  the short-history version underperforms for sample-size reasons, not
  because the variable lacks signal; this is the same honest constraint
  as the Customs flash feature, on a similar timeline to resolve itself.
- **Korea Customs flash export figures**: sourced and wired in
  (`src/customs_flash.py`, `src/run_layer1_customs.py`), directionally
  promising, but blocked on data depth rather than access now - the
  practical next step is finding a second source for pre-2021 flash
  figures (or waiting for more months to accumulate) to reach the
  36-month minimum needed to treat this as a real Layer result rather
  than an exploratory one.
- **Sector-wide earnings sentiment**: Layer 2 currently scores Samsung's
  guidance tone alone. Adding SK Hynix and Hyundai via DART's OpenDART
  API (same manual-key constraint) would test whether sentiment is a
  Samsung-specific artifact or a genuine sector-wide signal.
- **Volatility-regime uncertainty layer**: deliberately deferred. Before
  building anything, this needs an Engle ARCH-LM test on the SARIMA/blend
  residuals to check the series actually exhibits volatility clustering
  in the first place - in keeping with this project's rule that
  complexity has to earn its keep, not be assumed.

## Notebooks

- [01_layer0_baseline.ipynb](01_layer0_baseline.ipynb) - naive, seasonal naive, SARIMA
- [02_sarimax_exogenous.ipynb](02_sarimax_exogenous.ipynb) - SARIMAX with real exogenous data: the project's best point estimate, still unconfirmed
- [03_layer1_altdata.ipynb](03_layer1_altdata.ipynb) - structured alt-data, GBM vs. Ridge, SHAP
- [04_layer2_sentiment.ipynb](04_layer2_sentiment.ipynb) - Samsung earnings-call sentiment
- [05_customs_flash_exploratory.ipynb](05_customs_flash_exploratory.ipynb) - Customs flash feature: sourced for real, data-depth-limited, exploratory only
- [06_capstone_surface.ipynb](06_capstone_surface.ipynb) - feature-count x regularization surface, and the project's closing synthesis"""))

nb["cells"] = cells
nbf.write(nb, "00_summary.ipynb")
print("wrote 00_summary.ipynb")
