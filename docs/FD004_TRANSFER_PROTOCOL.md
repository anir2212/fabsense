# FD004 multi-condition transfer check: frozen before scoring

Written 1 November 2026 before loading FD004 cycle tables or labels, fitting a model for this check, or scoring an FD004 endpoint. Commit this protocol alone before implementation or evaluation. The pinned NASA archive's `readme.txt` was inspected for subset metadata: FD004 has **248 training engines, 249 test engines, six operating conditions and two fault modes**. No FD004 data row was inspected while choosing this design.

## Question and fixed comparison

The 23 October FD003 test found that an FD001-trained windowed MLP won FD001 development but lost to a fixed XGBoost baseline on FD003. Does that pattern persist when the same **FD001-trained** models encounter FD004's different operating conditions and fault mix? This is one bounded follow-up within NASA's simulator, chosen after the FD003 result was known. It cannot establish physical fleet or fab performance, and FD004 is not an independent source.

Use the existing FD001 training rows only, the 125-cycle training-target cap and predictions clipped to [0, 125]. Do not score FD001's official test or re-score FD003. Refit exactly:

1. `fabstack.cmapss.candidates()["xgboost"]`: 240 trees, depth 3, learning rate 0.05, subsample and column sample 0.8, seed 7, on the existing 109 causal rolling features with operating settings.
2. The already selected `mlp_window40_hidden128x64` from `evidence/cmapss_neural_fd003.json`, using `fabstack.cmapss_neural.mlp`, the same 14 sensors, window 40, seeds 0–4 and all 100 FD001 training engines. Do not search a new architecture or use FD004 to tune preprocessing. The MLP does not include operating settings, an existing model choice rather than a new ablation.
3. A training-median `DummyRegressor` as a descriptive floor.

Before reading FD004 tables, require the saved FD003 result's selection to be `mlp_window40_hidden128x64` and its FD001 development baseline RMSE to be 20.768165395590945 within 0.01. Require the saved result file SHA-256 `bf78043417a0f3ca328902e273186d4d3f04b8ac4532fe60466c393283197168` and the unchanged model implementation `fabstack/cmapss_neural.py` SHA-256 `d53d893332f59380fd5ec8ee3d9c3ba4f73b598b74540416f5c1d0ecd425643f`. The NASA archive SHA-256 is `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`. Changes to these inputs require a new protocol, not a silent rerun.

## Loader, scoring and reading rule

Add FD004 to the checked loader. Require 26 columns in each cycle table, 248 consecutive training engine IDs, 249 consecutive test engine IDs, one non-negative RUL label per test engine, no missing values and consecutive cycles within each engine. The FD004 training table is read **only** for those checks; it is never used to fit, scale, select or calibrate. No row count is fixed here because the archive readme reports engine counts, not cycle-row counts. After all models are refitted from FD001, score each FD004 test engine once at its last observed cycle against the official uncapped RUL label in ID order.

The primary descriptive comparison is **ΔRMSE = mean of the five MLP seed RMSEs minus XGBoost RMSE**, in cycles. Use the existing 2,000-resample paired engine bootstrap with seed 19, resampling the same engine indices for baseline and each MLP seed. An interval entirely above zero reads “MLP worse”; entirely below zero reads “MLP better”; otherwise “no demonstrated difference.” This interval covers fixed models and engine sampling in FD004, not training-seed, simulator or source uncertainty. Report all five seed RMSEs and the count beating XGBoost; do not hide an unfavorable seed mean.

Also report for baseline, each MLP seed and constant: RMSE, MAE, mean bias (prediction minus truth), NASA asymmetric score and engine-bootstrap intervals. Give fixed true-RUL slices **0–30**, **31–80** and **above 80** (counts, RMSE and bias); in the 0–30 slice, positive bias means overprediction near failure. These slices are descriptive and do not change the primary reading. Report the number of labels above the 125-cycle prediction cap. Do not infer a fault mode for any individual engine: the test files do not provide per-engine fault-mode labels.

Save aggregate results and provenance in `evidence/cmapss_fd004_transfer.json`, with the protocol commit/hash, archive hash, model versions, fixed configuration, training/validation data use, test-engine counts, seed rows, intervals, limits and any deviations. Raw NASA rows and per-engine predictions stay ignored. A second reproduction may compare saved metrics, but any post-result rerun or deviation must be disclosed. Do not rewrite the 23 October FD003 evidence or the earlier protocol.
