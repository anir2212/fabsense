# Neural remaining-life candidate: protocol, committed before evaluation

Written 23 October 2026, before any model has been fitted to or scored on C-MAPSS FD003. This file is committed on its own; the implementation and the FD003 result come in later commits, and the result section of `RESULTS.md` will cite this file's commit hash. Nothing below may change after the FD003 scores are produced. Any deviation will be reported as a deviation next to the result, not edited into this file.

## Why a new protocol is needed

The FD001 official test has already been scored once for the selected XGBoost model (17.28-cycle RMSE, 19 October 2026). Using it again to compare a new model would turn it into a second validation set. FD002, FD003 and FD004 are in the same pinned NASA archive (`data/raw/cmapss.zip`, inner `CMAPSSData.zip`) and, as of commit `78711d3`, no code, evidence or Markdown in this repository has read or mentioned them (checked with `grep -rIl "FD00[234]"` over the project and `git log --all -G 'FD00[234]'`, both empty). FD003 is used here as the untouched test.

FD003 has one operating condition like FD001, but two fault modes (HPC degradation and fan degradation) instead of one. It is therefore a **distribution shift**, not a second sample from FD001. A model trained only on FD001 may degrade on it, and that degradation is part of what is being measured.

## Development data

- **FD001 training engines only.** The existing seeded split in `fabstack/cmapss.py` (seed 7, 80 development engines, 20 validation engines) and the same 80 pseudo-endpoints (50%, 65%, 80% and 90% of each validation engine's life) are reused unchanged.
- **Targets:** remaining cycles within each engine, capped at **125** for training. Validation and test truths stay uncapped. All predictions are clipped to **[0, 125]**, as in the baseline.
- **Not used for anything:** the FD001 official test and its labels; FD003 training engines (loaded only for shape, ID and cycle checks, never for fitting, scaling, feature choice or selection); FD002 and FD004 (not loaded at all).

## Candidate set

1. **Baseline (fixed):** the existing selected model, XGBoost on the 109 causal rolling features from `causal_features`, with the same hyperparameters (`240` trees, depth `3`, learning rate `0.05`, subsample and column sample `0.8`, seed `7`). It is not re-tuned. Its recomputed development RMSE should reproduce the saved 20.77 cycles; a difference above 0.01 cycles will be reported.
2. **Neural candidate: an MLP over windowed features.** scikit-learn `MLPRegressor` (PyTorch is not installed and will not be installed). This is a plain feed-forward network on a flattened window, not an LSTM, CNN or any sequence model.
   - **Inputs:** the current cycle number plus the last *N* cycles of 14 sensors, flattened (`N × 14 + 1` columns). The sensors are 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20 and 21. They are the sensors with more than two distinct values in the FD001 training rows; sensors 1, 5, 6, 10, 16, 18 and 19 and the three operating settings are dropped. This list is fixed now from FD001 and is not revisited on FD003.
   - **Causality:** the window ends at the current cycle and resets at each engine. Lag *k* uses cycle *t − k* of the same engine; where *t − k* < 1 the engine's first cycle is repeated (edge padding). No later cycle and no other engine can enter a row.
   - **Pipeline:** `StandardScaler` on inputs and a standardised target (`TransformedTargetRegressor`), both fitted on the training rows of that fit only. `MLPRegressor` with ReLU, Adam, learning rate `1e-3`, batch size `256`, L2 `alpha = 1e-3`, `max_iter = 200`, `early_stopping = True` with `validation_fraction = 0.1` and `n_iter_no_change = 10`. Early stopping holds out a random 10% of the *training rows of that fit*; those rows come from development engines only, never from the 20 validation engines or FD003.
   - **Grid, listed in advance (4 configurations):** window *N* ∈ {20, 40} × hidden layers ∈ {(64,), (128, 64)}.
   - **Seeds:** each configuration is fitted with 5 seeds, `0, 1, 2, 3, 4` (the `random_state` for weight initialisation, batching and the early-stopping split). Every metric is reported per seed and as mean, standard deviation, minimum and maximum over the 5 seeds.

## Selection rule (FD001 development scores only)

1. **MLP configuration:** the configuration with the lowest **mean development RMSE over its 5 seeds** on the 80 pseudo-endpoints. Ties break toward the smaller window, then the smaller network.
2. **Protocol-selected model:** the MLP is selected over the baseline only if its 5-seed mean development RMSE is lower than the baseline's development RMSE. Otherwise the baseline stays selected.
3. After selection, the baseline and the chosen MLP configuration (all 5 seeds) are refitted on all 100 FD001 training engines. No FD003 value enters steps 1–3; the code computes FD003 features only after the refits.

## Held-out test: FD003

- Each of the 100 FD003 test engines is scored **once, at its last observed cycle**, against `RUL_FD003.txt` (uncapped, in engine-ID order).
- **Both** the baseline and the chosen MLP configuration are evaluated on FD003, whichever one was selected. A training-median constant (from capped FD001 targets) is also scored as a floor, as in the FD001 study. It is a reference, not a candidate.
- **Metrics, the same as FD001:** RMSE, MAE, mean bias (predicted − true), the NASA asymmetric score (sum and mean), the 2,000-resample engine-level bootstrap interval from `endpoint_uncertainty` (RMSE and mean score; for the MLP, one interval per seed), and RMSE by true-RUL slice (0–30, 31–80, above 80).
- **Paired comparison:** a 2,000-resample engine-level paired bootstrap (generator seed 19). Each resample draws one set of engine indices and applies it to the baseline and to all 5 MLP seeds. The statistic is ΔRMSE = (mean over seeds of MLP RMSE) − (baseline RMSE). Report the point value and the 2.5% and 97.5% percentiles. Seeds and fitted models are held fixed, so this interval covers engine sampling only.
- **Loader checks:** FD003 train `(24720, 26)`, test `(16596, 26)`, RUL `(100, 1)`; engine IDs 1–100 in both tables; no missing values; consecutive cycles within each engine; non-negative labels. These expected shapes come from published descriptions of the subset. If the files differ, the loader fails closed and the shapes are corrected only after checking the archive's `readme.txt`, before any model sees the data.

## Success and failure criteria, fixed now

- **Neural better on FD003:** the 95% paired interval for ΔRMSE lies entirely below 0.
- **Neural worse on FD003:** the interval lies entirely above 0.
- **No demonstrated difference:** the interval contains 0.
- **Seed robustness (descriptive):** the number of the 5 seeds whose own FD003 RMSE is below the baseline's.
- **The neural extension succeeds** only if the MLP is selected on development data **and** is "better on FD003". Every other combination is reported as a failure of the neural candidate to show an advantage, which is a valid result.
- **Baseline transfer:** the baseline is said to have **degraded under the FD003 shift** if its FD003 RMSE is above 19.83 cycles, the upper end of its FD001 official-test bootstrap interval. The MLP has no FD001 official-test number in this protocol, so its transfer is judged only relative to the baseline.
- **Floor:** a model that does not beat the constant reference's FD003 RMSE is reported as failing the floor.
- The FD003 RUL slices, bias and NASA score are descriptive and do not change any verdict above.

## What stays out

- **FD001 official test:** not re-run or re-tuned for either model. The MLP is not scored on it in this study. If an FD001 official number for the MLP is ever produced later, it must be labelled exploratory.
- **FD002 and FD004** (six operating conditions) stay untouched for later work, which will need its own committed protocol.
- **FD003 training engines** stay unused for fitting, so a future FD003-trained study remains possible, but FD003 test labels are no longer untouched after this run.
- Only one FD003 evaluation run is reported. If the script fails before producing any metrics (a bug, not a result), it may be fixed and rerun. If any rerun happens after metrics exist, every run is reported.

## Evidence to save

`evidence/cmapss_neural_fd003.json` will record the run date, hardware, Python and library versions, seeds, the protocol commit hash, per-seed development and FD003 metrics, the summaries above, the selection outcome and the verdicts. It holds aggregate numbers only, with no raw NASA rows.
