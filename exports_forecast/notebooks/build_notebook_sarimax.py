"""Generates 02_sarimax_exogenous.ipynb. Run once; the notebook itself is
the deliverable, this script is just how it's assembled."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# SARIMAX: Real Exogenous Data, Fit Jointly Rather Than Bolted On

## Introduction

Layer 0 set the baseline: SARIMA, fit on the exports series alone,
barely separates from a random walk. Before trying the more common
recipe of building lag features and alt-data and handing them to a
generic ML regressor (that's notebooks 03-04, and neither beats this
baseline either), this notebook asks the more direct question first -
can SARIMA's own structure be improved by giving it real exogenous data
directly (SARIMAX), fit as one coherent state-space model rather than as
lag features handed to a separate model?

Two new, real, freely-sourced exogenous variables, found by researching
what the trade-nowcasting literature and central banks actually use for
this problem, that this project didn't have before:

- **WSTS Asia Pacific semiconductor billings** - actual industry sales
  revenue, not a market-price proxy (unlike SOX/Samsung/SK Hynix stock
  prices already used in Layer 1). Free, no login, monthly back to 1986:
  https://www.wsts.org/67/Historical-Billings-Report. See
  `src/wsts_billings.py`.
- **IMF PortWatch Busan port trade tonnage** - satellite/AIS-tracked
  physical export volume through Korea's largest port. Free, no login,
  daily back to 2019-01-01, via the IMF's own production API (not a DIY
  proxy - this is the validated version of the "Layer 3: VIIRS nightlight"
  idea flagged as deprioritized in the README). See `src/portwatch.py`.

Both were confirmed live (real API calls, real downloaded files) before
any modeling code was written - the same discipline notebook 05 later
applies to the Customs flash data.

---
## Methodology

Two separate tests, because the two exogenous variables have very
different histories, and conflating them would confound "did the feature
help" with "did more training data help":

**Test 1 (`run_sarimax.py`):** SARIMA + both WSTS and Busan tonnage,
constrained to the 88 rows both variables share (Busan's 2019-01 start is
the binding constraint). Order selected with a capped grid (p,q,P,Q ≤ 1) -
an uncapped grid, tried first, produced numerically unstable, non-converged
forecasts (RMSE in the quadrillions) at the walk-forward origins with the
least training data, since a 6-parameter seasonal ARIMA structure isn't
safely identifiable on ~40-50 effective rows. Capping the grid fixed this
and is itself consistent with the project's Occam's-razor framing, not an
arbitrary patch.

**Test 2 (`run_sarimax_full.py`, the decisive one):** SARIMA + WSTS alone,
at Layer 0's actual training length - 436 rows, 1990-2026 - since WSTS's
own history (1986+) needs no truncation at all. This isolates the
exogenous-variable question from the sample-size question entirely: if
real industry data doesn't help SARIMA when SARIMA gets to use all the
history it normally would, that's a real answer about the feature.

Both tests are scored against the exact same 36-month held-out window as
every other model in this project (2023-05 to 2026-04) - confirmed
directly by checking that the naive forecast, which only depends on the
actual series values at each test date, is byte-identical across every
backtest CSV in this project's `data/` folder regardless of which script
produced it."""))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, "../src")

import pandas as pd
import matplotlib.pyplot as plt

results = {
    "naive": pd.read_csv("../data/backtest_naive.csv", index_col=0, parse_dates=True),
    "SARIMA (Layer 0)": pd.read_csv("../data/backtest_sarima.csv", index_col=0, parse_dates=True),
    "SARIMA+Ridge blend": pd.read_csv("../data/backtest_blend.csv", index_col=0, parse_dates=True),
    "SARIMAX + WSTS (full history)": pd.read_csv("../data/backtest_sarimax_wsts_full.csv", index_col=0, parse_dates=True),
    "SARIMAX + WSTS + Busan (short history)": pd.read_csv("../data/backtest_sarimax.csv", index_col=0, parse_dates=True),
    "SARIMA, same short history, no exog": pd.read_csv("../data/backtest_sarima_same_window.csv", index_col=0, parse_dates=True),
}

for name, df in results.items():
    print(f"{name:>40}  n={len(df):>3}  RMSE={((df['actual']-df['forecast'])**2).mean()**0.5:>14,.0f}  "
          f"MAPE={((df['actual']-df['forecast']).abs()/df['actual']).mean()*100:>6.2f}%")"""))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(11, 4))
results["SARIMA (Layer 0)"]["actual"].plot(ax=ax, label="actual", color="black", linewidth=2)
for name in ["SARIMA (Layer 0)", "SARIMAX + WSTS (full history)", "SARIMA+Ridge blend"]:
    results[name]["forecast"].plot(ax=ax, label=name, linestyle="--")
ax.set_title("36-month walk-forward: actual vs. forecast")
ax.set_ylabel("USD")
ax.legend()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Findings

**Test 2, full history, is the real result.** SARIMAX with WSTS Asia
Pacific billings added gets MAPE 2.50% / RMSE 2.17B - better than Layer
0's SARIMA (2.92% / 2.88B), better than naive (3.14%), and better than the
SARIMA+Ridge blend (2.65% / 2.38B) on both metrics. This is the best point
estimate anywhere in this project. A Diebold-Mariano test against SARIMA
gives p=0.17 - **not significant at 5%**, so this is reported with the
same discipline as the blend: a genuinely promising point estimate, not a
confirmed result. (Also not significant against naive, p=0.18, or against
the blend, p=0.34 - included for completeness, not because a comparison
against an already-unconfirmed model would prove anything on its own.)

What makes this a *more* interesting lead than the blend, even though
neither clears significance: the blend recombines two models already in
this project, while this result comes from genuinely new information -
real semiconductor industry billings, fit directly into the ARIMA
structure - and it's tested at full sample length with no sample-size
excuse available.

**Test 1, short history, shows the sample-size confound directly.**
Restricted to the 88 rows both WSTS and Busan tonnage share, plain SARIMA
(no exog) actually gets *worse* than naive (4.40% vs. 3.14% MAPE) purely
from losing ~350 months of training history. Adding the two exogenous
variables back on top of that same short history brings it to 4.03% - an
improvement over the short-history SARIMA, but still worse than naive, and
nowhere near Layer 0's full-history number. The lesson: on this series,
*how much history the model gets to see* matters more than either of the
two new variables did on their own in this test - which is exactly why
Test 2 (full history, WSTS alone) rather than Test 1 is the one that
actually moved the headline result.

Busan PortWatch tonnage's real constraint is the same shape as Customs
flash export data's: a genuinely relevant variable whose history (2019+)
is too short to fairly test yet, not a modeling failure. Unlike WSTS,
which happened to have decades of history because industry associations
have published billings reports since the 1980s, satellite AIS tracking
of ports is a 2019+ dataset by construction (that's when regular commercial
tracking coverage began) - there's no way to backfill more of it, only to
wait for more months to accumulate, the same honest limitation as
Customs flash.

## Key takeaways

The exact same 36-month test window has now been used to test six
different models across this project, and the two best point estimates
(this SARIMAX+WSTS result and the SARIMA+Ridge blend) both beat SARIMA,
and neither is statistically confirmed at this sample size. That
consistency across two structurally different approaches (a factor
combination vs. a genuinely new exogenous variable) is itself informative
- it suggests there may be real, small, hard-to-confirm-at-n=36 signal in
this series beyond univariate SARIMA, not that either result is a fluke,
but confirming that needs either a longer test window or a
pre-registered, single held-out confirmation period, not another look at
the same 36 months.

Real industry data (WSTS), fit directly into a proven statistical
structure (SARIMAX) rather than routed through a generic ML regressor on
top of lag features (Layers 1-2's approach), is the first alt-data attempt
in this project to produce a full-length, methodologically clean result
that beats Layer 0 on both RMSE and MAPE. Whether it's real signal or a
36-observation coincidence is exactly the question a longer evaluation
window would answer, and is honestly left open here rather than claimed
either way.

Next: notebooks 03-04 try the other, more common recipe - lag features
and alt-data through a GBM/Ridge - on this same series, to see whether a
more flexible model finds anything this direct, parsimonious approach
didn't."""))

nb["cells"] = cells
nbf.write(nb, "02_sarimax_exogenous.ipynb")
print("wrote 02_sarimax_exogenous.ipynb")
