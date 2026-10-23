# Measured local results

The SECOM section below uses attributed public UCI records; the two C-MAPSS sections (FD001, and FD001-trained models tested on FD003) use NASA's public **simulated turbofan** data. All later photo, MSA, facilities and question-router results use generated seeded data (`seed=7`) or authored development cases. None estimates performance on an external fab, aircraft fleet or building. Raw inputs and generated outputs are ignored by Git where stated; compact evidence is linked below.

## Neural remaining-life candidate tested once on C-MAPSS FD003, 23 October 2026

**Protocol, committed before evaluation:** [docs/NEURAL_RUL_PROTOCOL.md](docs/NEURAL_RUL_PROTOCOL.md), committed on its own before the implementation (file SHA-256 `55062eaf…434a`, recorded in the evidence with the development repository's commit IDs). `make cmapss-neural` was then run once, on 23 October 2026 at 11:08 UTC. Evidence: [`evidence/cmapss_neural_fd003.json`](evidence/cmapss_neural_fd003.json). There were no reruns and no deviations from the protocol. At run time two Markdown files had uncommitted wording edits, so the evidence records `uncommitted_tracked_changes_at_run: true`; no code or configuration differed from the committed implementation. Before the run, a loader unit test opened the real FD003 files to check their shapes, which the protocol allows; no model saw them.

**Setup.** Development used FD001 training engines only, with the same seeded 80/20 engine split, the same 80 pseudo-endpoints, the 125-cycle training cap and predictions clipped to 0–125. The FD001 official test was not scored for either model. The candidates were the existing selected XGBoost baseline (109 causal rolling features, unchanged settings) and a scikit-learn `MLPRegressor` over flattened windows of the last *N* cycles of 14 sensors plus the cycle number. It is a feed-forward network on windowed features, not an LSTM or CNN. The predeclared grid was *N* ∈ {20, 40} × hidden layers ∈ {(64), (128, 64)}, each with seeds 0–4. FD003 has one operating condition like FD001 but two fault modes, so it is a shifted test: **16,596** observed test rows from **100** truncated engines, with uncapped true RUL from 6 to 145 cycles (15 engines above the 125 cap). FD003 training rows were loaded only for the loader checks. FD002 and FD004 were not loaded.

**Development (FD001, 80 pseudo-endpoints from 20 held-out engines).** The recomputed baseline reproduced its saved 19 October validation RMSE exactly (20.77).

| Candidate | Input columns | Dev RMSE, 5-seed mean ± SD (min–max) | Dev MAE, mean | Dev asymmetric score, mean per case |
|---|---:|---:|---:|---:|
| XGBoost baseline (one fixed seed) | 109 | 20.77 | 14.62 | 14.78 |
| MLP, window 20, hidden 64 | 281 | 20.04 ± 0.73 (18.83–20.73) | 15.00 | 11.67 |
| MLP, window 20, hidden 128×64 | 281 | 19.99 ± 1.04 (18.74–21.50) | 14.42 | 11.80 |
| MLP, window 40, hidden 64 | 561 | 16.87 ± 0.69 (15.90–17.46) | 12.43 | 6.65 |
| MLP, window 40, hidden 128×64, **selected** | 561 | **16.64 ± 0.54** (16.20–17.48) | **11.97** | 6.71 |

By the predeclared rules the window-40, 128×64 MLP was the chosen configuration, and because its 16.64 mean development RMSE was below the baseline's 20.77, **the MLP was the protocol-selected model**. Every MLP fit stopped early (18–68 epochs) with no convergence warning.

**FD003 held-out test (100 engines, one score each at the last observed cycle).**

| Model (trained on all 100 FD001 training engines) | RMSE, cycles | MAE | Bias (pred − true) | Asymmetric score, total / per engine | Engine-bootstrap RMSE 95% |
|---|---:|---:|---:|---:|---:|
| XGBoost baseline | **27.23** | **21.59** | +12.31 | **4,681.8 / 46.8** | [23.44, 30.72] |
| MLP, 5-seed mean ± SD | 49.63 ± 0.75 | 33.15 ± 0.87 | +25.65 ± 1.03 | 999,177 / 9,991.8 (mean over seeds) | per-seed intervals all within [40.19, 59.02] |
| Training-median constant (reference) | 49.80 | 41.22 | +27.68 | 154,266 / 1,542.7 | – |

Per-seed MLP RMSE was 49.15, 49.41, 49.37, 50.96 and 49.26. The paired engine bootstrap gave ΔRMSE (MLP seed mean − baseline) = **+22.40 cycles, 95% interval [+15.15, +29.26]**.

**Verdicts under the committed criteria.**
- **Neural worse on FD003**: the paired interval lies entirely above zero, and none of the 5 seeds beat the baseline. The neural extension **did not succeed**: it was selected on development data and then lost clearly on the held-out subset.
- **The baseline degraded under the shift**: 27.23 RMSE against the 19.83 threshold (the top of its FD001 official-test interval).
- **Floor**: the baseline beat the constant reference (27.23 vs 49.80). The MLP's 5-seed mean was below it by only 0.17 cycles, and seed 3 (50.96) was above it. In practice the MLP was no better than a constant on FD003, and its asymmetric score was far worse, because its errors were large late predictions.

**Where the error came from (descriptive, not used for any decision).** Both models predicted too much remaining life for engines close to failure. For the 20 engines with true RUL 0–30, baseline RMSE was 30.56 (bias +25.5) and MLP RMSE was 87.6–88.9 across seeds (bias +70.9 to +72.7), so on average it predicted about 70 more cycles than these engines had left. Per-engine predictions are not saved, so the evidence shows these slice averages only. For the 33 engines at 31–80, baseline RMSE was 34.21 and MLP 44.3–46.9; for the 47 engines above 80, baseline 18.94 and MLP 18.9–24.9. One possible explanation is that FD003's second fault mode (fan degradation) moves some sensors in ways FD001 training never showed, and raw sensor windows extrapolate worse than XGBoost's tree splits. This run did not test that explanation; it would need per-fault-mode labels, which the test files do not include.

**What this does and does not show.** On FD001-like development engines a windowed MLP beat the tuned XGBoost by about 4 cycles of RMSE, but that ranking reversed on a subset with an extra fault mode. Development validation from one fault mode did not predict behaviour under this shift. The result is one simulator, one training subset and one shifted test. Intervals cover engine sampling with seeds and fitted models fixed; they do not cover the configuration choice or the simulator-to-real gap. FD003 test labels are no longer untouched. FD002 and FD004, with six operating conditions, remain unread. Nothing here is a measurement on fab equipment or an aircraft fleet.

**Run details.** Apple arm64, macOS 15.3.1, 11 logical CPUs, OpenBLAS limited to one thread; Python 3.12.13, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.1.1. Wall time 35.5 s for all 26 fits and both evaluations, peak resident memory about 1.2 GB. `make test` passed **48 tests** after this change (42 before); the 6 new tests cover the FD003 loader checks, window causality, seed determinism, the selection and verdict rules, and a run on synthetic engines showing that changing every FD003 value leaves development scores and selection unchanged. `make app-test` passed. The dashboard does not show the FD003 result.

## NASA C-MAPSS FD001 remaining-life baseline, 19 October 2026

NASA's [Prognostics Center of Excellence repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) supplies the archive. Its pinned SHA-256 and citation are in [data/SOURCES.md](data/SOURCES.md); the raw ZIP remains ignored. FD001 has **20,631** training cycle rows from 100 run-to-failure simulated engines, **13,096** observed test cycle rows from 100 other truncated engines, and 100 true remaining-cycle labels at their last observed test cycle. Each row has engine ID, cycle, three settings and 21 sensors. This is a simulation with one operating condition and one fault mode, not data from fab tools or an operating aircraft fleet.

The model uses **109 features**: current cycle/settings/sensors plus 5- and 20-cycle trailing sensor means and current-minus-mean differences. Rolling windows reset at each engine and use no future cycles. Training targets are capped at **125 cycles** to model the common early-life plateau; official test labels remain **uncapped** (observed range 7–145) for evaluation. A seeded engine-ID split held out 20 entire training engines. Four fixed life-fraction pseudo-endpoints from each gave 80 development validation cases. The following candidates were fixed before looking at official test scores; lowest validation RMSE selected XGBoost, then it was refitted on all 100 training engines.

| Candidate | Engine-held-out validation RMSE, cycles | Validation MAE, cycles | Validation asymmetric score, mean |
|---|---:|---:|---:|
| Training-row median | 56.66 | 49.10 | 1,293.62 |
| Ridge with train-fitted scaling | 22.60 | 17.86 | 14.32 |
| XGBoost, selected | **20.77** | **14.62** | 14.78 |

On the official **100-engine endpoint test**, selected XGBoost reached **17.28-cycle RMSE**, **12.71-cycle MAE**, and an asymmetric score of **488.38 total / 4.88 per engine**. A fixed training-median reference reached **49.82 RMSE** and **166,570.54 total asymmetric score**; it is a deliberately simple baseline, not the best possible age-aware predictor. The score uses `d = predicted RUL − actual RUL`, summing `exp(−d/13)−1` for early predictions (`d<0`) and `exp(d/10)−1` for late predictions; late predictions cost more, following the [PHM08 scoring formulation](https://www.papers.phmsociety.org/index.php/phmconf/article/download/2348/1323). A 2,000-resample **engine-level bootstrap** of the fixed test predictions gave a descriptive RMSE interval **[14.56, 19.83]** and mean-score interval **[3.25, 6.73]**. It does not include model-selection or simulator-to-real-world uncertainty.

Error varied by true remaining life: RMSE was **12.62** cycles for 25 engines at 0–30 RUL, **16.84** for 20 engines at 31–80, and **19.16** for 55 engines above 80. The evidence file [`evidence/cmapss_fd001.json`](evidence/cmapss_fd001.json) contains the exact engine split, validation scores, test metrics, interval and slices. The validation pseudo-endpoints are a fixed truncation scheme, not the official censoring mechanism. The official test was used once for the selected model and simple reference. As of 19 October, a neural RUL model, fault-onset detector, external fleet, calibration and prospective maintenance-cost policy remained open; the 23 October section above adds a predeclared MLP comparison on FD003 without re-scoring this official test.

## UCI SECOM yield baseline, 18 October 2026

The [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) has **1,567** rows, **590** numeric sensor columns, and **104** in-house test failures (`1`; pass is `-1`). The raw archive hash is pinned in `fabstack/secom.py` and [data/SOURCES.md](data/SOURCES.md). The timestamp in the label file ordered a single 80/20 split; it was excluded from model inputs. The first **1,253** rows contained **87** failures and were used for fixed-model, stratified five-fold development. The final **314** rows contained **17** failures and were read only for the final evaluation. Missingness filtering at 50%, median imputation with missing indicators, and near-zero-variance removal were fitted **inside each fold**, then fitted once on all development rows for the held-out test. Logistic regression used balancing and scaling; the forest used balanced subsamples; XGBoost used fold-local balanced sample weights.

| Candidate | Development CV AP mean ± SD | Development CV ROC AUC mean | Held-out AP | Held-out ROC AUC | Failures in top 16 risk-ranked test rows |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.1463 ± 0.0463 | 0.6255 | **0.2250** | **0.7897** | **4/16** |
| Random forest | 0.2094 ± 0.0524 | 0.7335 | 0.0614 | 0.4896 | 1/16 |
| XGBoost | **0.2105 ± 0.1008** | 0.7047 | 0.0890 | 0.5453 | 1/16 |

The test positive-rate average-precision baseline was **0.0541**. XGBoost was selected by development CV AP before inspecting the test; its test AP dropped to **0.0890**. Logistic regression happened to score highest on the test, but choosing it *after* seeing those scores would reuse the test for selection. A 50% precision operating threshold chosen from each candidate's development out-of-fold scores produced **zero test alerts** for all three candidates (confusion matrix TN 297, FP 0, FN 17, TP 0). Thus the experiment did not establish a useful high-precision automatic fail detector. At a fixed 5% review capacity (16 test rows), logistic ranked 4 failures (**25% precision, 23.5% recall**); forest and XGBoost each ranked one (**6.25% precision, 5.9% recall**). This is descriptive test analysis, not a threshold tuned for deployment.

XGBoost's native TreeSHAP contributions were computed on the held-out matrix; the top transformed feature was anonymous `sensor_059`. The full top-20 and all fold/test metrics are in [`evidence/secom_baseline.json`](evidence/secom_baseline.json). Attribution is to model log-odds, not a physical or causal explanation. The source is a single historical dataset; no independent site, prospective window, process expert, cost curve or uncertainty interval for the final test is available. Correlation pruning and the planned wafer-map/BDG2 modules remain open; FD001's separate simulated RUL baseline is above.

## Photo bay

`make photo` ran 42 scenarios (7 maximum train sizes × 3 arrival rates × 2 recipe mixes), 20 independent seeds each: **840 SimPy replications**. The stylised bay has two scanners and shared A/B/C reticles; an earliest-due queue dispatches lots. The scenario means have replication-based 95% t intervals in `runs/photo_grid.json` and `figures/photo_tradeoff.png`.

At 0.05 lots/min with a balanced mix, maximum train size 1 gave 69.8 lots/day and 341.9 minute mean p90 cycle time; size 4 gave 71.5/day and 103.3 minutes; size 8 gave about 71.5/day and 97.9 minutes. For the skewed mix, size 1 gave 63.5/day and 1,046.2 minutes, size 8 gave 71.5/day and 128.2 minutes, and size 24 gave 71.4/day and 143.2 minutes. At 0.08 lots/min, balanced size 24 reached 104.0/day but mean p90 cycle time remained 1,042.3 minutes: the modeled system is overloaded, so this is not a viable operating recommendation.

A gradient-boosting surrogate fitted on other replication seeds had holdout mean absolute error **2.785 lots/day** for throughput and **189.65 minutes** for p90 cycle time. The example utility is `throughput/day - 0.03*max(0,p90_cycle_minutes-180)`; its selections are illustrative and sensitive to the uncalibrated assumptions and p90 prediction error.

## Measurement system

`make msa` generated 180 readings: 10 parts × 3 operators × 3 repeats × 2 workstations. Crossed ANOVA estimated old/new gauge R&R at **7.626% / 3.284%** of total observed variation. Observed old/new bias was **0.1359 / 0.0246** against injected **0.15 / 0.03**. Bland–Altman new-minus-old mean was **−0.1113**, with limits **[−0.1529, −0.0696]**. Cpk was **0.651 / 0.688**, below a typical 1.33 decision example; capability here belongs to this generated distribution. EWMA and CUSUM first signaled at index **57** after a planted drift at 50. These are simulation checks, not workstation qualification.

## Facilities data and cited analytics

`make utilities` generated 20 buildings × 3 meter types × 14 days hourly, **20,160 rows**. The first 10 days (14,400 rows) trained a weather/time/building `HistGradientBoostingRegressor`; the final 4 days (5,760 rows) held 30 injected spike/drift/stuck anomaly rows. The fixed residual threshold yielded **28 TP, 0 FP, 2 FN**, precision **1.000**, recall **0.933** on those planted cases. This is a synthetic temporal holdout, not generalization to real buildings.

DuckDB generated 20 row-cited building/meter insights; `verify_insights` re-queried and checked **20/20** totals. The allowlisted SQL question router answered **23/23 supported** authored questions with source-row citations and abstained on **2/2 unsupported** cases. This evaluates deterministic routing on its development set; there is no LLM or independent semantic QA evaluation.

## Verification

On 19 October, `LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q` passed **19 tests**, including causal-feature, grouped-endpoint, score-asymmetry and archive-hash guards. After adding the separate FD001 evidence tab, `make app-test` passed; an in-process Streamlit inspection found **five tabs**, the official **100-engine / 17.28-cycle RMSE / 12.71-cycle MAE** metrics and no script exception. This is a local UI render check, not a browser, accessibility or production-server check. Figure files were visually inspected after generation. No Docker, Power BI, remote CI or cloud run is claimed.

### Dashboard review checks, 20 October 2026

No model or metric changed. Each dashboard tab now loads only its own evidence through `fabstack/dashboard_evidence.py`. A missing file shows a warning, and a malformed file an error; both name the file and the Make command that rebuilds it, and no values are shown from that file. The suite passed **38 tests** (`make test`), up from 19. They include AppTests for an empty artifact folder, a folder with only the committed FD001 evidence, malformed evidence, and the full project dashboard matching its saved JSON. `make app-test` passed.

`make browser-test` ran headless Google Chrome 155.0.8059.82 through Playwright 1.64.0 against Streamlit 1.50.0 on 127.0.0.1 (Python 3.12.13, macOS 15.3.1 arm64). It passed **25/25 checks** across two scenarios ([`evidence/browser_walkthrough.json`](evidence/browser_walkthrough.json)):
- **Full evidence.** All five tabs appear in order with their headers and no warning or exception. The nine displayed metrics equal the saved JSON (for example 30 injected anomaly rows, 7.63% → 3.28% %GRR, and 100 engines / 17.28 RMSE / 12.71 MAE), and so do the engine bootstrap interval and both engine tables. The question box can be reached and submitted by keyboard; it returned the cited answer for B09.
- **Empty evidence.** Every tab names its missing file and rebuild command, and no metric value is shown.
- **Keyboard behaviour.** In both scenarios, the arrow keys move selection and focus between tabs and wrap at both ends, the visible panel follows the selected tab, and exactly one tab has `tabindex=0`.

The first browser run found two problems, which were fixed before the passing run. Streamlit 1.50 ignored the request to hide the index in `st.table`, so a meaningless 0/1/2 row-number column appeared; the engine tables are now plain Markdown tables. The checker had also read the tables before the second one rendered; it now waits for both.

`make boundary-check` ([`evidence/public_data_boundary.json`](evidence/public_data_boundary.json)) found both locally present archives matching their pinned SHA-256: SECOM 1,964,989 bytes and C-MAPSS 12,429,152 bytes. The hashes agree across the code, `data/SOURCES.md` and the saved evidence. All four query IDs used are documented in `warehouse/QUERY_CATALOG.md` with none undocumented. The 20 insight totals re-verified, 23 router citations point to flagged rows, and 25/25 saved router answers replay identically. The in-process dashboard metrics equal the saved JSON. Nothing was downloaded.

These are local render and consistency checks. They are not a screen-reader or WCAG audit, a production deployment, or evidence that any result applies to real equipment.

### Contrast fixes and rescan, 22 October 2026

The app-level contrast findings below were fixed, and `make a11y-scan` was run again ([evidence](evidence/accessibility_scan.json)):
- `.streamlit/config.toml` now sets a light theme with text colour `#111111`, so faded captions and inactive tab labels pass AA, and accent colour `#B3261E`, because Streamlit's default red failed as the selected-tab label.
- Two captions no longer use inline code. Captions fade everything inside them, so inline code there could not reach the required contrast whatever its colour.

**Colour contrast now passes on every tab** in both the full and empty-evidence scenarios; it was failing on all ten tab views. Three rules still fail. One is an app-level `aria-allowed-attr` on a Streamlit widget's popup `div` (full-evidence Photo bay). The others are Streamlit's main menu ARIA attribute, its unnamed header button, and the Vega chart-actions `<summary>`, none of which the app controls. The Sustainability chart still has 29 contrast items axe could not decide. The browser walkthrough still passes 25/25, and 42 tests and the AppTest pass. No screen-reader pass has been done, so this remains a findings list rather than a rating.

### Automated accessibility scan, 20 October 2026 (evening)

`make a11y-scan` injected axe-core 4.12.1 into each of the five tabs and ran its WCAG 2.0/2.1 A and AA rules on the whole page ([`evidence/accessibility_scan.json`](evidence/accessibility_scan.json)). This used headless Chrome 155.0.8059.85, Streamlit 1.50.0 and Playwright 1.64.0, with the full and empty-evidence scenarios. Four rules failed. A selector/HTML heuristic split the offending elements into Streamlit's own page chrome and app content.

The main app-level problem is colour contrast (rated serious), on every tab. It comes from the grey `st.caption` lines (the subtitle, the artifact-root note and the tab captions, including inline code inside them) and from the inactive tab labels: 2–4 elements per tab with full evidence, and 5–10 in the empty state, where captions explain the missing files. The colours are Streamlit's default theme. The full-evidence Photo bay tab also has one `aria-allowed-attr` failure (critical), a `div` with `aria-haspopup` inside a Streamlit widget.

The rest is outside the app's own content. On every tab, Streamlit's main menu (`#MainMenu`) fails `aria-allowed-attr` and the unnamed header button `stBaseButton-headerNoPadding` fails `button-name` (critical). On the Sustainability tab, the Vega chart's actions menu has an unnamed `<summary>` (`summary-name`, serious).

Between 22 and 28 rules passed per tab. The Sustainability chart also produced 29 contrast items axe could not decide ("needs review"). No fixes were applied in this run. An automated scan finds only some problems, and no screen-reader pass was done, so this is a findings list, not an accessibility rating. 42 tests pass, including 4 new tests of the scan's pure helpers. Sample HTML in the evidence has local temporary and home paths replaced with `<local-path>`.
