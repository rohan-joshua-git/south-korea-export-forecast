# South Korea Monthly Total Exports: Forecasting

Q1 (Macro QR), NUS Investment Society recruitment case.

**Start with [`notebooks/00_summary.ipynb`](notebooks/00_summary.ipynb)** - a
one-page results table across every layer plus the key takeaways, generated
directly from the same backtest CSVs the per-layer notebooks below produce
(not hand-typed). Read it before the rest of this README or the layer
notebooks.

## Structure

- `notebooks/00_summary.ipynb` - results table across all layers, takeaways, portfolio framing, roadmap - start here
- `src/data.py` - pulls the target series (FRED `XTEXVA01KRM667S`) and caches it locally
- `src/walk_forward.py` - rolling-origin walk-forward backtest harness (univariate + exogenous-features variants), reused by every layer
- `src/baseline.py` - Layer 0 models: naive, seasonal naive, SARIMA
- `src/features.py` - Layer 1 alt-data loaders: USD/KRW, oil, SOX, Samsung, SK Hynix, BDRY (Baltic Dry proxy), Google Trends
- `src/customs_flash.py` - Korea Customs flash export figures, scraped from public press releases (data.go.kr API signup never worked out - see file for why and what was tried); 94/131 releases parse cleanly but only back to 2021-05
- `src/run_layer1_customs.py` - exploratory (not assignment-valid, see file) check of whether the customs flash feature helps, given its short real history
- `src/layer1_model.py` - GBM and Ridge forecasters, reused by Layers 1 and 2
- `src/earnings_sentiment.py` - Layer 2: downloads Samsung's quarterly earnings-call PDFs and scores them with the Loughran-McDonald dictionary
- `src/layer2_model.py` - Layer 1 features + Samsung sentiment
- `src/run_ensemble.py` - SARIMA + Ridge forecast combination, with honest out-of-sample weight selection and a Diebold-Mariano significance test
- `src/diebold_mariano.py` - DM test with the Harvey-Leybourne-Newbold small-sample correction
- `src/capstone_surface.py` - builds the RMSE vs. feature-count vs. regularization-strength grid
- `src/wsts_billings.py` - real global semiconductor billings (WSTS Blue Book), free, monthly back to 1986
- `src/portwatch.py` - satellite/AIS-tracked Busan port trade tonnage (IMF PortWatch), free, daily back to 2019-01
- `src/sarimax_model.py`, `src/run_sarimax.py`, `src/run_sarimax_full.py` - SARIMA with real exogenous data fit directly into the state-space model, rather than through a separate GBM/Ridge; the project's best point estimate, see notebook 02
- `notebooks/01_layer0_baseline.ipynb` - Layer 0 writeup
- `notebooks/02_sarimax_exogenous.ipynb` - SARIMAX with WSTS billings and Busan port data; best point estimate in the project, still unconfirmed
- `notebooks/03_layer1_altdata.ipynb` - Layer 1 writeup
- `notebooks/04_layer2_sentiment.ipynb` - Layer 2 writeup
- `notebooks/05_customs_flash_exploratory.ipynb` - Customs flash feature, sourced for real but data-depth-limited; explicitly exploratory, not a Layer
- `notebooks/06_capstone_surface.ipynb` - the feature-count x regularization surface, and the project's closing synthesis

Notebooks are ordered by model family rather than strict chronology: 01→02 is the SARIMA/SARIMAX line (the approach that worked), 03→04 is the alt-data-via-ML line (the approach that mostly didn't, with a live diagnosis), 05 is one more open lead, and 06 closes the project with a direct empirical answer to the case's own Occam's razor framing.

## Setup

```
pip install -r requirements.txt
```

## Layers (see the full plan for detail; ordered to match the notebooks)

- **Layer 0 (done):** SARIMA/naive baseline. SARIMA barely edges out a random walk (MAPE 2.92% vs. 3.14%) - a genuinely tough bar, not a strawman.
- **SARIMAX (done):** researched what the trade-nowcasting literature and central banks actually use for this problem, then fit real exogenous data (WSTS global semiconductor billings; IMF PortWatch's satellite-tracked Busan port trade volume - the validated version of the Layer 3 idea below) directly into SARIMA's structure, rather than through a separate GBM/Ridge on lag features the way Layers 1-2 did. WSTS has full 1986+ history; at Layer 0's actual training length (436 rows), **SARIMAX beats SARIMA, naive, and the blend on both RMSE and MAPE** (2.50% vs. SARIMA's 2.92%) - the best point estimate in the project. Still **not statistically significant** (DM p=0.17), reported with the same discipline as the blend. A second version with Busan tonnage added is constrained to 2019+ (88 rows) and reproduces the sample-size problem the Customs feature hit - see notebook 02 for the full comparison, including why an uncapped SARIMA order grid produced numerically unstable forecasts on that shorter window and had to be capped.
- **Layer 1 (done, first pass):** alt-data (SOX/Samsung/SK Hynix as memory-cycle proxy, BDRY as Baltic Dry proxy, USD/KRW, oil, Google Trends) → GBM and Ridge. **Neither beats Layer 0** (GBM MAPE 4.92%, Ridge 3.30%). Root cause looks like sample size (BDRY's 2018 start caps the dataset at 98 rows) more than the alt-data being useless - Ridge closes most of the gap that GBM opens by overfitting.
- **Layer 2 (done, first pass):** adds Samsung earnings-call sentiment (real PDFs from samsung.com/global/ir, scored with Loughran-McDonald). Ridge improves slightly (MAPE 3.30% → 3.26%), GBM gets marginally worse. **Still doesn't beat Layer 0.** SHAP ranks sentiment mid-pack - real signal, not dominant. SK Hynix/Hyundai not included (no scrapeable PDF archive; DART needs the same kind of manual API key as Customs).
- **Ensemble (done):** simple SARIMA + Ridge forecast average, weight chosen on a pre-test validation window (not the test set). Best point estimate in the project until SARIMAX above (MAPE 2.65% vs. SARIMA's 2.92%), **but the Diebold-Mariano test says this is not statistically significant at n=36** (p=0.15) - reported as a directionally promising, not confirmed, result.
- **Customs flash feature (explored, not adopted as a Layer):** sourced for real by scraping Korea Customs' public press releases after the data.go.kr API path fell through entirely (blocked signup, and the originally-targeted dataset turned out to be the wrong one anyway - see `src/customs_flash.py`). Parses cleanly for 94/131 releases, but real coverage only reaches back to 2021-05, leaving 50 usable rows once merged - well under the assignment's 36-month walk-forward minimum. An explicitly-labeled exploratory check on a 14-origin window (`src/run_layer1_customs.py`) shows both Ridge and GBM improving slightly with the feature added (Ridge MAPE 3.56% vs. 3.62% without, same window) - directionally the most promising single feature tried in this project, consistent with it being a genuine leading rather than coincident indicator, but **not reportable as a validated Layer result** given the sample size, so it's kept separate from the headline numbers rather than blended into Layer 1's official 98-row backtest.
- **Capstone (done):** OOS RMSE vs. feature-count vs. regularization-strength surface, built on Layer 2's fixed 96-row dataset with features added in SHAP order and Ridge's alpha gridded 0.1-1000. Confirms the bowl shape directly: more features need more regularization, and without it, more features actively hurt. The best single cell in the 80-cell grid (10 features, alpha=100, RMSE 2.41B) beats everything from Layers 0-2 and the blend, though not SARIMAX (2.17B) - flagged explicitly as a scan-the-grid artifact, not a validated result, since it wasn't chosen on a held-out window the way SARIMA's order and the blend's weight were. Placed last deliberately: it's the direct empirical answer to the case's own Occam's razor framing, and its closing section ties every earlier layer's result back to this one surface.
- **Layer 3 (superseded by the SARIMAX work above):** the original idea (VIIRS nightlight around Busan port) was deprioritized, then replaced with something better once research surfaced it: IMF PortWatch's satellite-tracked port trade data is a real, published, production version of the same concept, not a DIY proxy.

## Known environment quirks

- `lightgbm` segfaults intermittently on repeated `.fit()` calls in this
  Windows setup (reproducible native-library crash, unrelated to numpy
  version - tried both numpy 1.26 and 2.x). Layers 1-2 use sklearn's
  `HistGradientBoostingRegressor` instead, which is stable. Worth retrying
  lightgbm on Linux if training speed matters later.
- Installing `shap`/`lightgbm` pulled numpy to 2.x, which conflicts with
  `langchain` elsewhere on this machine (wants numpy<2). Didn't break
  anything in this project, but worth knowing about if other local projects
  depend on langchain - a project-local virtualenv would isolate this.
- `data/samsung_pdfs/*.pdf` (~58MB) are gitignored; the extracted `.txt`
  cache is kept since that's what's actually reproducible from source-controlled
  code without re-downloading. Same pattern for `data/customs_pdfs/*.pdf`/`*.hwp`.
- `src/customs_flash.py` shells out to the `hwp5txt` CLI (installed via the
  `pyhwp` package in requirements.txt) to read `.hwp` attachments - confirm
  `pyhwp` installed cleanly and `hwp5txt` is on PATH before relying on it;
  only needed for releases that don't attach a PDF.

## Closing

Six notebooks, one running standard: a finding only counts once it clears
an honest out-of-sample bar, not because it looks good on the page. By
that standard, most of what got tried here - alt-data through GBM/Ridge,
earnings sentiment, the Customs flash feature at its current sample size
- didn't clear it, and is reported as exactly that, not reframed as a
near miss. One thing did: SARIMAX with real semiconductor billings data,
the first result in this project to beat SARIMA on the full, valid
36-month test the case asks for. Even that is held to the same bar and
reported as an unconfirmed point estimate, not a proven edge, because a
Diebold-Mariano test says it should be. If this project demonstrates one
thing, it's less any single number and more that the discipline held in
both directions - complexity had to earn its keep everywhere it was
tried, and when something finally did earn it, that got checked just as
hard as everything that didn't.
