# Measured local results

All results below use generated seeded data (`seed=7`) or authored development cases. They do **not** estimate performance on external fab or building data. Generated raw outputs are ignored by Git and reproducible with `make all`.

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

`LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **12 passed**. `Streamlit AppTest` rendered all four tabs, six headline metrics and no exception. Figure files were visually inspected after generation. No Docker, Power BI, remote CI or cloud run is claimed.
