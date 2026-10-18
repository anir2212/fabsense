"""Engine-separated FD001 remaining-life baseline on NASA's simulated turbofans.

The official test trajectories are truncated. Development choices use only
run-to-failure training engines; the official endpoint labels are scored once.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
ARCHIVE = Path("data/raw/cmapss.zip")
RESULT = Path("evidence/cmapss_fd001.json")
SEED = 7
RUL_CAP = 125
SENSORS = [f"sensor_{index:02d}" for index in range(1, 22)]
COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3", *SENSORS]
VALIDATION_FRACTIONS = (0.5, 0.65, 0.8, 0.9)


def ensure_archive(path: Path = ARCHIVE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("NASA C-MAPSS archive hash mismatch")
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("NASA C-MAPSS archive hash mismatch")
    return path


def load_fd001(path: Path = ARCHIVE) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    with zipfile.ZipFile(ensure_archive(path)) as outer:
        nested = [name for name in outer.namelist() if name.endswith("/CMAPSSData.zip")]
        if len(nested) != 1:
            raise ValueError("Expected one nested CMAPSSData.zip")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as data:
            train = pd.read_csv(io.BytesIO(data.read("train_FD001.txt")), sep=r"\s+", header=None)
            test = pd.read_csv(io.BytesIO(data.read("test_FD001.txt")), sep=r"\s+", header=None)
            rul = pd.read_csv(io.BytesIO(data.read("RUL_FD001.txt")), sep=r"\s+", header=None)
    if train.shape != (20631, 26) or test.shape != (13096, 26) or rul.shape != (100, 1):
        raise ValueError("Unexpected FD001 table shape")
    for frame in (train, test):
        frame.columns = COLUMNS
        if frame.isna().any().any() or set(frame["unit"]) != set(range(1, 101)):
            raise ValueError("Missing values or unexpected FD001 unit IDs")
        expected = frame.groupby("unit")["cycle"].cumcount().to_numpy() + 1
        if not np.array_equal(frame["cycle"].to_numpy(), expected):
            raise ValueError("FD001 cycles must be ordered and consecutive within each unit")
    labels = rul.iloc[:, 0].to_numpy(dtype=int)
    if np.any(labels < 0):
        raise ValueError("Negative official RUL label")
    return train, test, labels


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the current and earlier cycles of each engine, never future cycles."""
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle rows")
    if not frame.sort_values(["unit", "cycle"]).index.equals(frame.index):
        raise ValueError("Rows must be ordered by engine and cycle")
    features = frame[["cycle", "setting_1", "setting_2", "setting_3", *SENSORS]].copy()
    for window in (5, 20):
        mean = (
            frame.groupby("unit", sort=False)[SENSORS]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .reindex(frame.index)
        )
        for sensor in SENSORS:
            features[f"{sensor}_mean_{window}"] = mean[sensor]
            features[f"{sensor}_delta_{window}"] = frame[sensor] - mean[sensor]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite FD001 feature")
    return features


def targets(frame: pd.DataFrame) -> np.ndarray:
    final_cycle = frame.groupby("unit")["cycle"].transform("max")
    return (final_cycle - frame["cycle"]).to_numpy(dtype=float)


def validation_indices(frame: pd.DataFrame, units: list[int]) -> list[int]:
    """One pseudo-truncated endpoint at each fixed life fraction per held-out engine."""
    indices: list[int] = []
    for unit in units:
        engine = frame.loc[frame["unit"] == unit]
        lifetime = int(engine["cycle"].iloc[-1])
        for fraction in VALIDATION_FRACTIONS:
            cutoff = max(1, round(lifetime * fraction))
            indices.append(int(engine.loc[engine["cycle"] == cutoff].index[0]))
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate validation endpoints")
    return indices


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """NASA's asymmetric penalty makes late predictions costlier than early ones."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("RUL arrays must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RUL arrays must be finite")
    error = predicted - actual
    penalty = np.expm1(np.where(error < 0, -error / 13, error / 10))
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias_pred_minus_actual": float(np.mean(error)),
        "nasa_score_sum": float(np.sum(penalty)),
        "nasa_score_mean": float(np.mean(penalty)),
    }


def candidates() -> dict:
    from xgboost import XGBRegressor

    return {
        "constant_median": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "xgboost": XGBRegressor(
            n_estimators=240, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            tree_method="hist", random_state=SEED, n_jobs=4,
        ),
    }


def bounded_prediction(model, features: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(features), dtype=float), 0, RUL_CAP)


def endpoint_uncertainty(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Resample official test engines; intervals do not include model-selection error."""
    rng = np.random.default_rng(19)
    rmse, nasa_mean = [], []
    for _ in range(2000):
        sample = rng.integers(0, len(actual), len(actual))
        metrics = score(actual[sample], predicted[sample])
        rmse.append(metrics["rmse"])
        nasa_mean.append(metrics["nasa_score_mean"])
    return {
        "method": "2000 engine-level bootstrap resamples of the fixed test predictions",
        "rmse_95": np.quantile(rmse, [0.025, 0.975]).tolist(),
        "nasa_score_mean_95": np.quantile(nasa_mean, [0.025, 0.975]).tolist(),
    }


def run(path: Path = ARCHIVE) -> dict:
    train, test, official_rul = load_fd001(path)
    train_features = causal_features(train)
    raw_targets = targets(train)
    capped_targets = np.minimum(raw_targets, RUL_CAP)
    rng = np.random.default_rng(SEED)
    engine_order = rng.permutation(np.arange(1, 101))
    development_units = sorted(engine_order[:80].tolist())
    validation_units = sorted(engine_order[80:].tolist())
    development_mask = train["unit"].isin(development_units).to_numpy()
    endpoint_indices = validation_indices(train, validation_units)
    validation_actual = raw_targets[endpoint_indices]
    model_specs = candidates()
    validation = {}
    for name, model in model_specs.items():
        model.fit(train_features.loc[development_mask], capped_targets[development_mask])
        validation[name] = score(
            validation_actual,
            bounded_prediction(model, train_features.iloc[endpoint_indices]),
        )
    selected = min(validation, key=lambda name: (validation[name]["rmse"], name))

    # The official endpoint labels have not influenced the choice above.
    final_model = candidates()[selected]
    final_model.fit(train_features, capped_targets)
    test_features = causal_features(test)
    last_rows = test.groupby("unit", sort=True).tail(1)
    endpoint_features = test_features.loc[last_rows.index]
    if last_rows["unit"].tolist() != list(range(1, 101)):
        raise ValueError("Official test endpoint order differs from RUL label order")
    predictions = bounded_prediction(final_model, endpoint_features)
    baseline = DummyRegressor(strategy="median").fit(train_features, capped_targets)
    baseline_predictions = bounded_prediction(baseline, endpoint_features)
    slices = {}
    for name, mask in {
        "rul_0_to_30": official_rul <= 30,
        "rul_31_to_80": (official_rul > 30) & (official_rul <= 80),
        "rul_above_80": official_rul > 80,
    }.items():
        slices[name] = score(official_rul[mask], predictions[mask])
    return {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": SHA256,
        "dataset": "NASA PCoE C-MAPSS FD001, simulated turbofans",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_engines": 100,
        "test_engines": 100,
        "feature_count": len(train_features.columns),
        "training_target_cap_cycles": RUL_CAP,
        "official_test_labels_uncapped": True,
        "validation_endpoint_fractions": list(VALIDATION_FRACTIONS),
        "development_engine_ids": development_units,
        "validation_engine_ids": validation_units,
        "validation_endpoint_count": len(endpoint_indices),
        "validation": validation,
        "selection_rule": "lowest engine-held-out pseudo-endpoint RMSE; alphabetical tie-break",
        "selected_model": selected,
        "official_test": score(official_rul, predictions),
        "official_test_ci_95": endpoint_uncertainty(official_rul, predictions),
        "official_test_rul_slices": slices,
        "official_test_constant_median": score(official_rul, baseline_predictions),
        "test_truth_min_max": [int(official_rul.min()), int(official_rul.max())],
        "test_prediction_min_max": [float(predictions.min()), float(predictions.max())],
        "scope": "one simulated FD001 split; one endpoint per official test engine; no physical fleet or fault-onset validation",
    }


def main() -> None:
    result = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "selected_model", "official_test", "official_test_constant_median"
    )}, indent=2))


if __name__ == "__main__":
    main()
