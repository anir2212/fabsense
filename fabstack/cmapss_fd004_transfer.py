"""Frozen FD001-trained model check on NASA's simulated FD004 subset."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyRegressor

from fabstack import cmapss, cmapss_neural

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/FD004_TRANSFER_PROTOCOL.md"
PROTOCOL_V2 = ROOT / "docs/FD004_TRANSFER_PROTOCOL_V2.md"
PRIOR = ROOT / "evidence/cmapss_neural_fd003.json"
MODEL_CODE = ROOT / "fabstack/cmapss_neural.py"
RESULT = ROOT / "evidence/cmapss_fd004_transfer.json"
RUN = ROOT / "runs/cmapss_fd004_transfer.json"
PROTOCOL_SHA256 = "2162c43112c1a50b08917acb1b98d5d6ff62e909bd62debe228365aa98f0f4b2"
PROTOCOL_V2_SHA256 = "e6bff60ad726732e4e17cfdb8e2422a3496420c93b1c73b0cb3323ea0147a1c5"
PRIOR_SHA256 = "bf78043417a0f3ca328902e273186d4d3f04b8ac4532fe60466c393283197168"
MODEL_SHA256 = "d53d893332f59380fd5ec8ee3d9c3ba4f73b598b74540416f5c1d0ecd425643f"
SEEDS = (0, 1, 2, 3, 4)
WINDOW = 40
HIDDEN = (128, 64)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard() -> dict:
    for path, expected in ((PROTOCOL, PROTOCOL_SHA256), (PROTOCOL_V2, PROTOCOL_V2_SHA256),
                           (PRIOR, PRIOR_SHA256),
                           (MODEL_CODE, MODEL_SHA256)):
        if digest(path) != expected:
            raise RuntimeError(f"frozen FD004 input changed: {path.name}")
    cmapss.ensure_archive(ROOT / cmapss.ARCHIVE)
    prior = json.loads(PRIOR.read_text())
    selected = prior["selection"]
    if (selected["mlp_config_selected"] != "mlp_window40_hidden128x64"
            or abs(selected["baseline_development_rmse"] - 20.768165395590945) > 0.01
            or selected["protocol_selected_model"] != "mlp"):
        raise RuntimeError("FD003 development selection differs from frozen protocol")
    return prior


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        **cmapss.score(actual, predicted),
        "ci_95": cmapss.endpoint_uncertainty(actual, predicted),
        "rul_slices": cmapss_neural.slice_scores(actual, predicted),
    }


def run() -> dict:
    guard()
    started = datetime.now(UTC)
    # The FD001 loader reads its official tables for integrity; only train rows enter
    # fitting. No FD001/FD003 official label is used in this study.
    train, _fd001_test, _fd001_labels = cmapss.load_fd001(ROOT / cmapss.ARCHIVE)
    del _fd001_test, _fd001_labels
    actual_train = np.minimum(cmapss.targets(train), cmapss.RUL_CAP)
    baseline_features = cmapss.causal_features(train)
    window_features = cmapss_neural.window_features(train, WINDOW)
    baseline, baseline_fit = cmapss_neural._fit(
        cmapss.candidates()["xgboost"], baseline_features, actual_train
    )
    mlps = [
        (seed, *cmapss_neural._fit(cmapss_neural.mlp(HIDDEN, seed), window_features,
                                  actual_train))
        for seed in SEEDS
    ]
    constant = DummyRegressor(strategy="median").fit(baseline_features, actual_train)

    # This is the first FD004 table read; the model choices and refits are complete.
    fd004_train, test, labels = cmapss.load_subset("FD004", ROOT / cmapss.ARCHIVE)
    del fd004_train  # loader validation only; never used to fit or select
    endpoints = cmapss_neural._endpoints(test)
    if len(endpoints) != 248 or len(labels) != len(endpoints):
        raise ValueError("FD004 endpoint and label count must both be 248")
    baseline_test = cmapss.causal_features(test).loc[endpoints]
    window_test = cmapss_neural.window_features(test, WINDOW).loc[endpoints]
    truth = labels.astype(float)
    baseline_pred = cmapss.bounded_prediction(baseline, baseline_test)
    constant_pred = cmapss.bounded_prediction(constant, baseline_test)
    seed_pred = np.vstack([
        cmapss.bounded_prediction(model, window_test) for _, model, _ in mlps
    ])
    baseline_result = _metrics(truth, baseline_pred)
    constant_result = _metrics(truth, constant_pred)
    mlp_rows = [{"seed": seed, **_metrics(truth, predicted)}
                for (seed, _, _), predicted in zip(mlps, seed_pred)]
    paired = cmapss_neural.paired_delta_bootstrap(truth, baseline_pred, seed_pred)
    measurements = {
        "fd004_test_rows": len(test),
        "fd004_test_engines": len(endpoints),
        "truth_min_max": [int(labels.min()), int(labels.max())],
        "truth_above_cap": int((truth > cmapss.RUL_CAP).sum()),
        "baseline": baseline_result,
        "mlp": {"config": "mlp_window40_hidden128x64", "per_seed": mlp_rows,
                "mean_rmse": float(np.mean([row["rmse"] for row in mlp_rows]))},
        "constant_median_reference": constant_result,
        "paired_comparison": paired,
        "reading": cmapss_neural.verdict(paired["interval_95"]),
        "seeds_beating_baseline": sum(row["rmse"] < baseline_result["rmse"] for row in mlp_rows),
    }
    result = {
        "observed_at_utc": started.isoformat(),
        "protocol_commit": "f9ed5bc54da42151f7875b749c727713b09ae97d",
        "protocol_sha256": digest(PROTOCOL),
        "pre_score_correction_commit": "5c1d7de50498906f0914287a055fbeeeb1db3e6d",
        "pre_score_correction_sha256": digest(PROTOCOL_V2),
        "source_sha256": cmapss.SHA256,
        "prior_fd003_evidence_sha256": digest(PRIOR),
        "model_code_sha256": digest(MODEL_CODE),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "machine": platform.machine()},
        "data_use": "FD001 training engines fitted; FD001/FD003 official tests not scored; FD004 train validated only; FD004 test endpoints scored once",
        "training": {"baseline": baseline_fit,
                     "mlp": [{"seed": seed, **fit} for seed, _, fit in mlps]},
        "measurements": measurements,
        "scope": "one NASA simulated FD004 subset, six conditions and two fault modes; no per-engine fault labels, physical fleet or fab validation",
    }
    if RESULT.exists():
        saved = json.loads(RESULT.read_text())
        result["matches_saved_measurements"] = measurements == saved["measurements"]
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    result = run()
    print(json.dumps({"measurements": result["measurements"],
                      "matches_saved_measurements": result.get("matches_saved_measurements")},
                     indent=2))


if __name__ == "__main__":
    main()
