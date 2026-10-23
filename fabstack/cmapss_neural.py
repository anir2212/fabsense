"""Predeclared MLP-over-windowed-features RUL candidate, tested once on C-MAPSS FD003.

The protocol in docs/NEURAL_RUL_PROTOCOL.md was committed before this module
existed. Development uses FD001 training engines only; FD003 test endpoints are
read after both models are selected and refitted. The FD001 official test is not
scored here. This is a feed-forward network on a flattened window, not an LSTM
or CNN.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fabstack import cmapss

RESULT = Path("evidence/cmapss_neural_fd003.json")
PROTOCOL = Path("docs/NEURAL_RUL_PROTOCOL.md")
SAVED_FD001 = Path("evidence/cmapss_fd001.json")
# Sensors with more than two distinct values in FD001 training rows (fixed in the protocol).
WINDOW_SENSORS = [f"sensor_{index:02d}" for index in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
# Listing order is the tie-break order: smaller window first, then smaller network.
GRID = [(window, hidden) for window in (20, 40) for hidden in ((64,), (128, 64))]
SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_SEED = 19
BOOTSTRAP_RESAMPLES = 2000
BASELINE = "xgboost"
METRICS = ("rmse", "mae", "bias_pred_minus_actual", "nasa_score_sum", "nasa_score_mean")
SLICES = {
    "rul_0_to_30": (None, 30),
    "rul_31_to_80": (30, 80),
    "rul_above_80": (80, None),
}


def window_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Current cycle plus the last `window` cycles of each sensor, from the same engine only.

    Lag k at cycle t is the reading at cycle t-k; before the engine's first cycle the
    first reading is repeated. No later cycle and no other engine can enter a row.
    """
    if window < 1:
        raise ValueError("Window must cover at least the current cycle")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle rows")
    if not frame.sort_values(["unit", "cycle"]).index.equals(frame.index):
        raise ValueError("Rows must be ordered by engine and cycle")
    groups = frame.groupby("unit", sort=False)[WINDOW_SENSORS]
    first = groups.transform("first")
    columns = {"cycle": frame["cycle"].astype(float)}
    for lag in range(window):
        shifted = groups.shift(lag).fillna(first)
        for sensor in WINDOW_SENSORS:
            columns[f"{sensor}_lag_{lag:02d}"] = shifted[sensor]
    features = pd.DataFrame(columns, index=frame.index)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite window feature")
    return features


def mlp(hidden: tuple[int, ...], seed: int) -> TransformedTargetRegressor:
    network = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-3,
        batch_size=256, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10, random_state=seed,
    )
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), network), transformer=StandardScaler()
    )


def config_name(window: int, hidden: tuple[int, ...]) -> str:
    return f"mlp_window{window}_hidden{'x'.join(map(str, hidden))}"


def summarize(per_seed: list[dict]) -> dict:
    summary = {}
    for metric in METRICS:
        values = np.array([scores[metric] for scores in per_seed], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def select_config(mean_rmse: dict[str, float], order: list[str]) -> str:
    """Lowest 5-seed mean development RMSE; ties go to the earlier grid entry."""
    return min(order, key=lambda name: (mean_rmse[name], order.index(name)))


def choose_model(baseline_rmse: float, mlp_mean_rmse: float) -> str:
    return "mlp" if mlp_mean_rmse < baseline_rmse else BASELINE


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((predicted - actual) ** 2, axis=-1))


def paired_delta_bootstrap(actual: np.ndarray, baseline: np.ndarray, seeds: np.ndarray) -> dict:
    """ΔRMSE = mean over seeds of MLP RMSE − baseline RMSE, resampling engines jointly."""
    actual = np.asarray(actual, dtype=float)
    seeds = np.atleast_2d(np.asarray(seeds, dtype=float))
    point = float(_rmse(actual, seeds).mean() - _rmse(actual, baseline))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_RESAMPLES)
    for draw in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(actual), len(actual))
        deltas[draw] = (
            _rmse(actual[sample], seeds[:, sample]).mean()
            - _rmse(actual[sample], baseline[sample])
        )
    return {
        "statistic": "mean over seeds of MLP RMSE minus baseline RMSE, cycles",
        "method": f"{BOOTSTRAP_RESAMPLES} paired engine-level bootstrap resamples; seeds and models fixed",
        "point": point,
        "interval_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def verdict(interval: list[float]) -> str:
    low, high = interval
    if high < 0:
        return "neural better"
    if low > 0:
        return "neural worse"
    return "no demonstrated difference"


def slice_scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for name, (low, high) in SLICES.items():
        mask = np.ones(len(actual), dtype=bool)
        if low is not None:
            mask &= actual > low
        if high is not None:
            mask &= actual <= high
        if mask.any():
            result[name] = cmapss.score(actual[mask], predicted[mask])
    return result


def _fit(model, features: pd.DataFrame, target: np.ndarray) -> tuple[object, dict]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, target)
    info = {
        "fit_seconds": round(time.perf_counter() - started, 2),
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
    }
    if isinstance(model, TransformedTargetRegressor):
        network = model.regressor_[-1]
        info["epochs"] = int(network.n_iter_)
        info["best_internal_validation_r2"] = float(network.best_validation_score_)
    return model, info


def _endpoints(test: pd.DataFrame) -> pd.Index:
    last_rows = test.groupby("unit", sort=True).tail(1)
    if last_rows["unit"].tolist() != sorted(set(test["unit"])):
        raise ValueError("Test endpoint order differs from RUL label order")
    return last_rows.index


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def provenance() -> dict:
    import sklearn
    import xgboost

    return {
        "run_date_utc": datetime.now(UTC).isoformat(),
        "protocol_file": str(PROTOCOL),
        "protocol_commit": _git("log", "-1", "--format=%H", "--", str(PROTOCOL)),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() if PROTOCOL.exists() else None,
        "code_commit_at_run": _git("rev-parse", "HEAD"),
        "uncommitted_tracked_changes_at_run": _git("status", "--short", "--untracked-files=no") != "",
        "hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "thread_limits": {
                key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
            },
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }


def run(
    load_development: Callable = cmapss.load_fd001,
    load_test: Callable = lambda: cmapss.load_subset("FD003"),
    grid: list[tuple[int, tuple[int, ...]]] = GRID,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    # Development: FD001 training engines only. The official FD001 test is discarded unread.
    train, _official_test, _official_labels = load_development()
    del _official_test, _official_labels
    raw_targets = cmapss.targets(train)
    capped_targets = np.minimum(raw_targets, cmapss.RUL_CAP)
    development_units, validation_units = cmapss.engine_split()
    development_mask = train["unit"].isin(development_units).to_numpy()
    endpoint_indices = cmapss.validation_indices(train, validation_units)
    validation_actual = raw_targets[endpoint_indices]

    baseline_features = cmapss.causal_features(train)
    baseline, baseline_fit = _fit(
        cmapss.candidates()[BASELINE],
        baseline_features.loc[development_mask], capped_targets[development_mask],
    )
    baseline_validation = cmapss.score(
        validation_actual,
        cmapss.bounded_prediction(baseline, baseline_features.iloc[endpoint_indices]),
    )

    windows = {window: window_features(train, window) for window in sorted({w for w, _ in grid})}
    order = [config_name(window, hidden) for window, hidden in grid]
    development = {}
    for (window, hidden), name in zip(grid, order):
        features = windows[window]
        per_seed = []
        for seed in seeds:
            model, info = _fit(
                mlp(hidden, seed), features.loc[development_mask], capped_targets[development_mask]
            )
            scores = cmapss.score(
                validation_actual, cmapss.bounded_prediction(model, features.iloc[endpoint_indices])
            )
            per_seed.append({"seed": seed, **scores, **info})
        development[name] = {
            "window": window,
            "hidden_layers": list(hidden),
            "input_columns": len(features.columns),
            "per_seed": per_seed,
            "summary": summarize(per_seed),
        }
    mean_rmse = {name: development[name]["summary"]["rmse"]["mean"] for name in order}
    chosen = select_config(mean_rmse, order)
    selected = choose_model(baseline_validation["rmse"], mean_rmse[chosen])
    chosen_window, chosen_hidden = grid[order.index(chosen)]

    # Refit both models on all 100 FD001 training engines before any FD003 row is read.
    baseline_final, baseline_refit = _fit(
        cmapss.candidates()[BASELINE], baseline_features, capped_targets
    )
    mlp_final = []
    for seed in seeds:
        model, info = _fit(mlp(chosen_hidden, seed), windows[chosen_window], capped_targets)
        mlp_final.append((seed, model, info))
    constant = DummyRegressor(strategy="median").fit(baseline_features, capped_targets)

    # Held-out test: FD003, one score per engine at its last observed cycle.
    _test_train, test, test_labels = load_test()
    del _test_train  # loaded for the loader's shape, ID and cycle checks only
    endpoints = _endpoints(test)
    if len(endpoints) != len(test_labels):
        raise ValueError("Endpoint count differs from RUL label count")
    test_baseline_features = cmapss.causal_features(test).loc[endpoints]
    test_window_features = window_features(test, chosen_window).loc[endpoints]
    actual = test_labels.astype(float)
    baseline_pred = cmapss.bounded_prediction(baseline_final, test_baseline_features)
    constant_pred = cmapss.bounded_prediction(constant, test_baseline_features)
    seed_preds = np.vstack([
        cmapss.bounded_prediction(model, test_window_features) for _, model, _ in mlp_final
    ])
    mlp_per_seed = []
    for (seed, _, info), predicted in zip(mlp_final, seed_preds):
        mlp_per_seed.append({
            "seed": seed,
            **cmapss.score(actual, predicted),
            "ci_95": cmapss.endpoint_uncertainty(actual, predicted),
            "rul_slices": slice_scores(actual, predicted),
            "refit": info,
        })
    baseline_test = cmapss.score(actual, baseline_pred)
    constant_test = cmapss.score(actual, constant_pred)
    comparison = paired_delta_bootstrap(actual, baseline_pred, seed_preds)
    mlp_mean_rmse = float(np.mean([row["rmse"] for row in mlp_per_seed]))
    neural_verdict = verdict(comparison["interval_95"])
    saved_rmse = None
    if SAVED_FD001.exists():
        saved_rmse = json.loads(SAVED_FD001.read_text())["validation"][BASELINE]["rmse"]

    return {
        **provenance(),
        "dataset": "NASA PCoE C-MAPSS, simulated turbofans: FD001 train for development, FD003 test held out",
        "source_sha256": cmapss.SHA256,
        "data_use": {
            "development": "FD001 training engines only (seeded 80/20 engine split, 80 pseudo-endpoints)",
            "fd001_official_test": "not scored in this study",
            "fd003_train": "loaded for shape, ID and cycle checks only; not used for fitting, scaling or selection",
            "fd003_test": "scored once per engine at its last observed cycle after selection and refit",
            "fd002_fd004": "not loaded",
        },
        "training_target_cap_cycles": cmapss.RUL_CAP,
        "prediction_bounds_cycles": [0, cmapss.RUL_CAP],
        "window_sensors": WINDOW_SENSORS,
        "seeds": list(seeds),
        "grid": [{"name": n, "window": w, "hidden_layers": list(h)} for (w, h), n in zip(grid, order)],
        "development_engine_count": len(development_units),
        "validation_engine_ids": validation_units,
        "validation_endpoint_count": len(endpoint_indices),
        "development": {
            "baseline": {
                "model": BASELINE,
                "feature_count": len(baseline_features.columns),
                **baseline_validation,
                "fit": baseline_fit,
                "saved_fd001_validation_rmse": saved_rmse,
                "reproduces_saved_within_0_01": (
                    saved_rmse is not None and abs(saved_rmse - baseline_validation["rmse"]) <= 0.01
                ),
            },
            "mlp": development,
        },
        "selection": {
            "mlp_config_rule": "lowest 5-seed mean development RMSE; ties to smaller window, then smaller network",
            "mlp_config_selected": chosen,
            "model_rule": "MLP only if its 5-seed mean development RMSE is below the baseline's",
            "baseline_development_rmse": baseline_validation["rmse"],
            "mlp_development_rmse_mean": mean_rmse[chosen],
            "protocol_selected_model": selected,
        },
        "fd003_test": {
            "test_rows": len(test),
            "test_engines": len(endpoints),
            "truth_min_max": [int(test_labels.min()), int(test_labels.max())],
            "truth_above_cap": int((test_labels > cmapss.RUL_CAP).sum()),
            "baseline": {
                **baseline_test,
                "ci_95": cmapss.endpoint_uncertainty(actual, baseline_pred),
                "rul_slices": slice_scores(actual, baseline_pred),
                "prediction_min_max": [float(baseline_pred.min()), float(baseline_pred.max())],
                "refit": baseline_refit,
            },
            "mlp": {
                "config": chosen,
                "per_seed": mlp_per_seed,
                "summary": summarize(mlp_per_seed),
                "prediction_min_max": [float(seed_preds.min()), float(seed_preds.max())],
            },
            "constant_median_reference": constant_test,
            "paired_comparison": comparison,
        },
        "verdicts": {
            "fd003_comparison": neural_verdict,
            "seeds_beating_baseline": int(sum(row["rmse"] < baseline_test["rmse"] for row in mlp_per_seed)),
            "neural_extension_succeeds": selected == "mlp" and neural_verdict == "neural better",
            "baseline_degraded_threshold_rmse": 19.83,
            "baseline_degraded_under_shift": baseline_test["rmse"] > 19.83,
            "baseline_beats_constant_floor": baseline_test["rmse"] < constant_test["rmse"],
            "mlp_seed_mean_beats_constant_floor": mlp_mean_rmse < constant_test["rmse"],
        },
        "scope": (
            "simulated turbofans; FD001-trained models scored once on FD003 test endpoints "
            "(one condition, two fault modes); no physical fleet, fab equipment or onset validation"
        ),
    }


def main() -> None:
    result = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "selection": result["selection"],
        "fd003_baseline_rmse": result["fd003_test"]["baseline"]["rmse"],
        "fd003_mlp_rmse": result["fd003_test"]["mlp"]["summary"]["rmse"],
        "paired_comparison": result["fd003_test"]["paired_comparison"],
        "verdicts": result["verdicts"],
    }, indent=2))


if __name__ == "__main__":
    main()
