# FD004 transfer check: pre-score archive-count correction

Written 1 November 2026. This file supersedes only the FD004 **train/test engine-count and label-count assertions** in [`FD004_TRANSFER_PROTOCOL.md`](FD004_TRANSFER_PROTOCOL.md); every model, data-use, metric, interval, slice, reading and evidence rule in that separately committed protocol remains fixed. Both files must be read together for the complete design.

The first execution under the original protocol committed as `f9ed5bc` refitted the FD001 models, then failed at FD004 table-shape validation. It produced **no FD004 predictions, metrics or evidence file**. The original protocol had used the NASA archive's `readme.txt`, which says “Train trjectories: 248” and “Test trajectories: 249” for FD004. Inspection after the validation failure found the **pinned archive files themselves contain the reverse**:

| File | Observed shape | Engine IDs |
|---|---:|---:|
| `train_FD004.txt` | 61,249 × 26 | 1–249 |
| `test_FD004.txt` | 41,214 × 26 | 1–248 |
| `RUL_FD004.txt` | 248 × 1 | one label for each test engine |

The loader must now require **249 training engines, 248 test engines and 248 non-negative test RUL labels**, with exactly the row and column shapes above. The same missing-value, ID-order and consecutive-cycle checks remain. Score only the 248 test endpoints. FD004 training rows remain validation-only and are never used to fit, scale, select or calibrate. This correction is committed before the first successful score, and both the failed pre-score attempt and this correction must be reported with the result. The pinned outer archive hash is unchanged: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
