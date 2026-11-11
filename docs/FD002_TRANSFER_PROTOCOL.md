# FD002 fixed-model transfer check: frozen before table access

Written 2 November 2026 before reading FD002 cycle tables or labels, fitting a model for this check, or scoring an FD002 endpoint. Commit this protocol alone before implementation or evaluation. FD002 is described by NASA as a six-operating-condition, one-fault-mode simulation; no FD002 data row or label informed this design. The pinned outer archive is `data/raw/cmapss.zip`, SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.

## Question and fixed comparison

FD001-trained XGBoost beat a development-selected windowed MLP on FD003 and FD004. Does the same fixed comparison hold on the still-unread FD002 subset, where operating conditions change in the same simulator? This follow-up was chosen after those results; it is not an independent external validation or a new model selection. The training target remains capped at 125 cycles, predictions clipped to [0, 125], and official test labels uncapped.

Refit on **FD001 training engines only**, with no FD002 fitting, scaling, selection or calibration:

1. `fabstack.cmapss.candidates()["xgboost"]`, using the existing 109 causal rolling features including operating settings and the unchanged seed-7 hyperparameters.
2. The FD001-development-selected `mlp_window40_hidden128x64` from `evidence/cmapss_neural_fd003.json`: the existing 14-sensor flattened window, seeds 0–4, unchanged model and preprocessing code. The MLP omits operating settings, an existing model choice rather than a newly tuned ablation.
3. A training-median `DummyRegressor` on the XGBoost features as a descriptive reference.

Require the FD003 selection to remain `mlp_window40_hidden128x64`, its baseline development RMSE to remain 20.768165395590945 within 0.01, and these pre-score input hashes:

| Input | SHA-256 |
|---|---|
| `fabstack/cmapss.py` | `5fcdb56af9ad0ef4b42c1ce08232896b2e0a9f6d9836fdfb47a0772ec7341a99` |
| `fabstack/cmapss_neural.py` | `d53d893332f59380fd5ec8ee3d9c3ba4f73b598b74540416f5c1d0ecd425643f` |
| `evidence/cmapss_neural_fd003.json` | `bf78043417a0f3ca328902e273186d4d3f04b8ac4532fe60466c393283197168` |
| `evidence/cmapss_fd004_transfer.json` | `9e48860070b509655546d107c5b851b2fed5a73f2e0daa475d3bd35ee3765e33` |

Do not reread or rescore the FD001, FD003 or FD004 official test endpoints. Changes to the frozen model or prior evidence require a new protocol.

## Input checks and evaluation

After refitting the models, load FD002's `train`, `test` and `RUL` text files from the pinned archive. Require 26 columns in both cycle tables, one nonnegative official RUL label per test engine, no missing or nonfinite values, engine IDs consecutive from 1 in file order, cycles consecutive from 1 within each engine, and one last observed test cycle per engine. Record exact row/engine counts from the files. Do **not** fix a count from the archive readme: its FD004 train/test counts were previously reversed. If any integrity check fails, stop before endpoint predictions; disclose a separately committed pre-score correction if needed. FD002 training rows are read for validation only.

Score each FD002 test engine once at its last observed cycle. The primary descriptive comparison is **ΔRMSE = mean of the five MLP seed RMSEs minus XGBoost RMSE**, in cycles. Use the existing 2,000-resample paired engine bootstrap with seed 19, resampling the same test-engine indices for XGBoost and every MLP seed. An interval wholly above zero reads “MLP worse”; wholly below zero reads “MLP better”; otherwise “no demonstrated difference.” The interval describes these fixed models and FD002 engine sampling, not simulator, source or training-seed uncertainty.

Report XGBoost, all five MLP seeds and constant: RMSE, MAE, prediction-minus-truth bias, NASA asymmetric score and 95% engine-bootstrap intervals. Report the count of MLP seeds beating XGBoost, labels above the 125-cycle cap, and predeclared true-RUL slices **0–30**, **31–80** and **above 80** (counts, RMSE and bias). In the 0–30 slice, positive bias is overprediction near failure. Compare FD002's pattern with dated FD003 and FD004 values only descriptively; do not infer a per-engine fault mode or a causal operating-condition effect from cross-subset differences.

Save aggregate result and provenance in `evidence/cmapss_fd002_transfer.json`, and a first successful copy if the runner or provenance format later changes. Store the protocol commit/hash, source and frozen-input hashes, library versions, fixed model configuration, test-engine counts, seed rows, intervals and limits. Keep raw NASA tables and per-engine predictions ignored. Reproduce the measured block in a second execution and disclose any post-result change. Do not rewrite earlier evidence or protocols.
