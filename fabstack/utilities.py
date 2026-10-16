"""Synthetic weather-adjusted meter anomalies with row-level ground truth."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

METRICS = ("electricity", "chilled_water", "water")


@dataclass(frozen=True)
class Reading:
    row_id: int
    building: str
    metric: str
    hour_index: int
    hour_of_day: int
    weekday: int
    temperature_c: float
    observed: float
    anomaly_type: str | None = None
    predicted: float | None = None
    flagged: bool = False


def generate(seed: int = 7, buildings: int = 20, days: int = 14) -> list[Reading]:
    if buildings < 2 or days < 14:
        raise ValueError("need at least two buildings and 14 days")
    rng = np.random.default_rng(seed)
    readings = []
    for building in range(buildings):
        for metric_index, metric in enumerate(METRICS):
            scale = (90, 55, 18)[metric_index] * (0.8 + 0.025 * building)
            for hour_index in range(days * 24):
                hour = hour_index % 24
                weekday = (hour_index // 24) % 7
                temp = 28 + 4 * math.sin(hour_index / 24 * 2 * math.pi / 7) + rng.normal(0, 0.4)
                occupancy = 1.0 if weekday < 5 and 8 <= hour < 18 else 0.65
                expected = scale * (0.65 + 0.012 * temp + 0.35 * occupancy)
                observed = max(0.0, expected + rng.normal(0, scale * 0.015))
                readings.append(Reading(len(readings), f"B{building:02d}", metric,
                                        hour_index, hour, weekday, float(temp), float(observed)))
    # Thirty labelled holdout anomalies: ten spikes, ten rising drift values,
    # and two five-hour flatline intervals on distinct meters.
    index = {(row.building, row.metric, row.hour_index): offset
             for offset, row in enumerate(readings)}
    targets = []
    for step in range(10):
        targets.append((f"B{step:02d}", METRICS[step % 3], 270 + step * 3, "spike"))
        targets.append((f"B{step + 10:02d}", METRICS[(step + 1) % 3], 275 + step * 3, "drift"))
    for building, metric, start in (("B00", "electricity", 295),
                                    ("B01", "chilled_water", 310)):
        previous = readings[index[building, metric, start - 1]].observed
        for hour_index in range(start, start + 5):
            offset = index[building, metric, hour_index]
            readings[offset] = replace(readings[offset], observed=previous,
                                       anomaly_type="stuck")
    for building, metric, hour_index, kind in targets:
        offset = index[building, metric, hour_index]
        row = readings[offset]
        if row.anomaly_type:
            raise AssertionError("anomaly target collision")
        factor = 1.6 if kind == "spike" else 1.2 + 0.02 * (hour_index % 5)
        readings[offset] = replace(row, observed=row.observed * factor,
                                   anomaly_type=kind)
    assert sum(row.anomaly_type is not None for row in readings) == 30
    return readings


def _features(rows: list[Reading]) -> np.ndarray:
    return np.array([[int(row.building[1:]), METRICS.index(row.metric), row.hour_of_day,
                      row.weekday, row.temperature_c] for row in rows], dtype=float)


def detect(rows: list[Reading], train_days: int = 10) -> tuple[list[Reading], dict]:
    cutoff = train_days * 24
    train = [row for row in rows if row.hour_index < cutoff]
    test = [row for row in rows if row.hour_index >= cutoff]
    if not train or not test or any(row.anomaly_type for row in train):
        raise ValueError("need clean training history and a holdout")
    model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31,
                                          learning_rate=0.08, random_state=7)
    model.fit(_features(train), [row.observed for row in train])
    predicted_train = model.predict(_features(train))
    residuals = np.abs(np.array([row.observed for row in train]) - predicted_train)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    predicted_test = model.predict(_features(test))
    updated = [replace(row, predicted=float(prediction))
               for row, prediction in zip(test, predicted_test, strict=True)]
    # Flatline checks are per meter; 3+ identical values are unlikely in noisy data.
    streaks: dict[tuple[str, str], tuple[float, int]] = {}
    flagged = []
    for row in updated:
        key = (row.building, row.metric)
        previous, count = streaks.get(key, (float("nan"), 0))
        count = count + 1 if row.observed == previous else 1
        streaks[key] = (row.observed, count)
        threshold = max(5 * mad, 0.10 * row.predicted)
        residual_flag = abs(row.observed - row.predicted) > threshold
        flagged.append(replace(row, flagged=bool(residual_flag or count >= 3)))
    tp = sum(row.flagged and row.anomaly_type is not None for row in flagged)
    fp = sum(row.flagged and row.anomaly_type is None for row in flagged)
    fn = sum(not row.flagged and row.anomaly_type is not None for row in flagged)
    report = {"train_rows": len(train), "holdout_rows": len(test),
              "injected_anomaly_rows": tp + fn, "true_positive": tp,
              "false_positive": fp, "false_negative": fn,
              "precision": tp / (tp + fp) if tp + fp else 0.0,
              "recall": tp / (tp + fn) if tp + fn else 0.0,
              "train_absolute_residual_mad": mad,
              "model": "HistGradientBoostingRegressor on building/metric/hour/weekday/temp"}
    return flagged, report


def records(rows: list[Reading]) -> list[dict]:
    return [asdict(row) for row in rows]
