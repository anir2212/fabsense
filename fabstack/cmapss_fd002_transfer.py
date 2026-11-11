"""Frozen FD001-trained model check on NASA's simulated FD002 subset."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyRegressor

from fabstack import cmapss, cmapss_neural

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/FD002_TRANSFER_PROTOCOL.md"
PRIOR = ROOT / "evidence/cmapss_neural_fd003.json"
FD004 = ROOT / "evidence/cmapss_fd004_transfer.json"
RESULT = ROOT / "evidence/cmapss_fd002_transfer.json"
RUN = ROOT / "runs/cmapss_fd002_transfer.json"
PROTOCOL_SHA256 = "4bc2b1aef4353e90b5e354e31a70bf47510a2ccf834ffe6f1983f8eb091b2f45"
PROTOCOL_COMMIT = "b6579c7e14432c6f14c33b08e0f1aaec5f3897a6"
FROZEN_SHA256 = {
    "fabstack/cmapss.py": "5fcdb56af9ad0ef4b42c1ce08232896b2e0a9f6d9836fdfb47a0772ec7341a99",
    "fabstack/cmapss_neural.py": "d53d893332f59380fd5ec8ee3d9c3ba4f73b598b74540416f5c1d0ecd425643f",
    "evidence/cmapss_neural_fd003.json": "bf78043417a0f3ca328902e273186d4d3f04b8ac4532fe60466c393283197168",
    "evidence/cmapss_fd004_transfer.json": "9e48860070b509655546d107c5b851b2fed5a73f2e0daa475d3bd35ee3765e33",
}
SEEDS = (0, 1, 2, 3, 4)
WINDOW = 40
HIDDEN = (128, 64)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard() -> dict:
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen FD002 protocol changed")
    for rel, expected in FROZEN_SHA256.items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"frozen FD002 input changed: {rel}")
    cmapss.ensure_archive(ROOT / cmapss.ARCHIVE)
    prior = json.loads(PRIOR.read_text())
    selected = prior["selection"]
    if (selected["mlp_config_selected"] != "mlp_window40_hidden128x64"
            or abs(selected["baseline_development_rmse"] - 20.768165395590945) > 0.01
            or selected["protocol_selected_model"] != "mlp"):
        raise RuntimeError("FD003 development selection differs from frozen protocol")
    return prior


def validate_fd002(
    train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Check the unknown FD002 counts without trusting the archive readme."""
    if (train.empty or test.empty or train.shape[1] != 26 or test.shape[1] != 26
            or rul.shape != (test.iloc[:, 0].nunique(), 1)):
        raise ValueError("Unexpected FD002 table shape or label count")
    for name, frame in (("train", train), ("test", test)):
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"FD002 {name} has nonfinite values")
        frame.columns = cmapss.COLUMNS
        ids = frame["unit"].to_numpy(dtype=float)
        cycles = frame["cycle"].to_numpy(dtype=float)
        if not np.array_equal(ids, ids.astype(int)) or not np.array_equal(cycles, cycles.astype(int)):
            raise ValueError(f"FD002 {name} IDs or cycles are nonintegral")
        count = frame["unit"].nunique()
        if (set(frame["unit"]) != set(range(1, count + 1))
                or not frame["unit"].is_monotonic_increasing):
            raise ValueError(f"FD002 {name} unit IDs must be ordered and consecutive")
        expected_cycles = frame.groupby("unit")["cycle"].cumcount().to_numpy() + 1
        if not np.array_equal(frame["cycle"].to_numpy(), expected_cycles):
            raise ValueError(f"FD002 {name} cycles must start at 1 and be consecutive")
    label_values = rul.iloc[:, 0].to_numpy(dtype=float)
    if (not np.isfinite(label_values).all() or (label_values < 0).any()
            or not np.array_equal(label_values, label_values.astype(int))):
        raise ValueError("FD002 RUL labels must be finite nonnegative integers")
    return train, test, label_values.astype(int)


def load_fd002(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    with zipfile.ZipFile(cmapss.ensure_archive(path)) as outer:
        nested = [name for name in outer.namelist() if name.endswith("/CMAPSSData.zip")]
        if len(nested) != 1:
            raise ValueError("Expected one nested CMAPSSData.zip")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as data:
            frames = [pd.read_csv(io.BytesIO(data.read(f"{kind}_FD002.txt")),
                                  sep=r"\s+", header=None)
                      for kind in ("train", "test", "RUL")]
    return validate_fd002(*frames)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {**cmapss.score(actual, predicted),
            "ci_95": cmapss.endpoint_uncertainty(actual, predicted),
            "rul_slices": cmapss_neural.slice_scores(actual, predicted)}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def run() -> dict:
    guard()
    started = datetime.now(UTC)
    train, _fd001_test, _fd001_labels = cmapss.load_fd001(ROOT / cmapss.ARCHIVE)
    del _fd001_test, _fd001_labels
    target = np.minimum(cmapss.targets(train), cmapss.RUL_CAP)
    baseline_features = cmapss.causal_features(train)
    window_features = cmapss_neural.window_features(train, WINDOW)
    baseline, baseline_fit = cmapss_neural._fit(
        cmapss.candidates()["xgboost"], baseline_features, target)
    mlps = [(seed, *cmapss_neural._fit(cmapss_neural.mlp(HIDDEN, seed),
                                      window_features, target)) for seed in SEEDS]
    constant = DummyRegressor(strategy="median").fit(baseline_features, target)

    # The first FD002 table read happens only after all model refits.
    fd002_train, test, labels = load_fd002(ROOT / cmapss.ARCHIVE)
    source_shapes = {"train_rows": len(fd002_train),
                     "train_engines": fd002_train["unit"].nunique(),
                     "test_rows": len(test), "test_engines": test["unit"].nunique(),
                     "labels": len(labels)}
    del fd002_train  # validation only
    endpoints = cmapss_neural._endpoints(test)
    if len(endpoints) != len(labels):
        raise ValueError("FD002 endpoints do not align with labels")
    baseline_test = cmapss.causal_features(test).loc[endpoints]
    window_test = cmapss_neural.window_features(test, WINDOW).loc[endpoints]
    truth = labels.astype(float)
    baseline_pred = cmapss.bounded_prediction(baseline, baseline_test)
    constant_pred = cmapss.bounded_prediction(constant, baseline_test)
    seed_pred = np.vstack([cmapss.bounded_prediction(model, window_test)
                           for _, model, _ in mlps])
    baseline_result = _metrics(truth, baseline_pred)
    constant_result = _metrics(truth, constant_pred)
    mlp_rows = [{"seed": seed, **_metrics(truth, prediction)}
                for (seed, _, _), prediction in zip(mlps, seed_pred)]
    paired = cmapss_neural.paired_delta_bootstrap(truth, baseline_pred, seed_pred)
    measurements = {
        "source_shapes": source_shapes,
        "truth_min_max": [int(labels.min()), int(labels.max())],
        "truth_above_cap": int((truth > cmapss.RUL_CAP).sum()),
        "baseline": baseline_result,
        "mlp": {"config": "mlp_window40_hidden128x64", "per_seed": mlp_rows,
                "mean_rmse": float(np.mean([row["rmse"] for row in mlp_rows]))},
        "constant_median_reference": constant_result,
        "paired_comparison": paired,
        "reading": cmapss_neural.verdict(paired["interval_95"]),
        "seeds_beating_baseline": sum(row["rmse"] < baseline_result["rmse"]
                                    for row in mlp_rows),
    }
    result = {
        "observed_at_utc": started.isoformat(),
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": digest(PROTOCOL),
        "runner_commit_at_run": _git_head(),
        "source_sha256": cmapss.SHA256,
        "frozen_inputs_sha256": FROZEN_SHA256,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "machine": platform.machine()},
        "data_use": "FD001 training engines fitted; FD001/FD003/FD004 official tests not scored; FD002 train validated only; FD002 test endpoints scored",
        "training": {"baseline": baseline_fit,
                     "mlp": [{"seed": seed, **fit} for seed, _, fit in mlps]},
        "measurements": measurements,
        "scope": "one NASA simulated FD002 subset; no independent physical-fleet or fab validation",
    }
    if RESULT.exists():
        saved = json.loads(RESULT.read_text())
        result["matches_saved_measurements"] = measurements == saved["measurements"]
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    result = run()
    measurements = result["measurements"]
    print(json.dumps({"source_shapes": measurements["source_shapes"],
                      "baseline_rmse": measurements["baseline"]["rmse"],
                      "mlp_mean_rmse": measurements["mlp"]["mean_rmse"],
                      "paired_comparison": measurements["paired_comparison"],
                      "reading": measurements["reading"],
                      "matches_saved_measurements": result.get("matches_saved_measurements")},
                     indent=2))


if __name__ == "__main__":
    main()
