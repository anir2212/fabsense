"""Crossed Gauge R&R ANOVA, bias, agreement, capability and SPC diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import linregress


@dataclass(frozen=True)
class Measurement:
    workstation: str
    part: int
    operator: int
    repeat: int
    true_value: float
    measured: float


STATIONS = {
    "old": {"bias": 0.15, "repeatability_sd": 0.06, "operator_effect": 0.04,
            "interaction_sd": 0.02},
    "new": {"bias": 0.03, "repeatability_sd": 0.03, "operator_effect": 0.02,
            "interaction_sd": 0.01},
}


def generate(seed: int = 7, parts: int = 10, operators: int = 3,
             repeats: int = 3) -> list[Measurement]:
    if min(parts, operators, repeats) < 2:
        raise ValueError("crossed ANOVA needs at least two levels per factor")
    rng = np.random.default_rng(seed)
    true_values = np.linspace(8.5, 11.5, parts) + rng.normal(0, 0.03, parts)
    rows = []
    for workstation, config in STATIONS.items():
        operator_offsets = np.linspace(-config["operator_effect"],
                                       config["operator_effect"], operators)
        interactions = rng.normal(0, config["interaction_sd"], (parts, operators))
        for part in range(parts):
            for operator in range(operators):
                for repeat in range(repeats):
                    measured = (true_values[part] + config["bias"] +
                                operator_offsets[operator] + interactions[part, operator] +
                                rng.normal(0, config["repeatability_sd"]))
                    rows.append(Measurement(workstation, part, operator, repeat,
                                            float(true_values[part]), float(measured)))
    return rows


def capability(values: list[float], lower: float, upper: float) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or lower >= upper or values.std(ddof=1) == 0:
        raise ValueError("valid limits and nonconstant values required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    return {"mean": mean, "sample_sd": sd, "cp": (upper - lower) / (6 * sd),
            "cpk": min(upper - mean, mean - lower) / (3 * sd)}


def crossed_anova(rows: list[Measurement]) -> dict:
    parts = sorted({row.part for row in rows})
    operators = sorted({row.operator for row in rows})
    repeats = sorted({row.repeat for row in rows})
    p, o, r = len(parts), len(operators), len(repeats)
    if min(p, o, r) < 2 or len(rows) != p * o * r:
        raise ValueError("balanced full-crossed part/operator/repeat design required")
    cells = {(row.part, row.operator, row.repeat): row.measured for row in rows}
    if len(cells) != len(rows):
        raise ValueError("duplicate or missing crossed cell")
    array = np.array([[[cells[(part, operator, repeat)] for repeat in repeats]
                       for operator in operators] for part in parts])
    grand = array.mean()
    by_part = array.mean(axis=(1, 2))
    by_operator = array.mean(axis=(0, 2))
    by_cell = array.mean(axis=2)
    ss_part = o * r * np.square(by_part - grand).sum()
    ss_operator = p * r * np.square(by_operator - grand).sum()
    ss_interaction = r * np.square(by_cell - by_part[:, None] -
                                   by_operator[None, :] + grand).sum()
    ss_repeat = np.square(array - by_cell[:, :, None]).sum()
    ms_part = ss_part / (p - 1)
    ms_operator = ss_operator / (o - 1)
    ms_interaction = ss_interaction / ((p - 1) * (o - 1))
    ms_repeat = ss_repeat / (p * o * (r - 1))
    var_repeat = float(ms_repeat)
    var_interaction = max(0.0, float((ms_interaction - ms_repeat) / r))
    var_operator = max(0.0, float((ms_operator - ms_interaction) / (p * r)))
    var_part = max(0.0, float((ms_part - ms_interaction) / (o * r)))
    var_grr = var_repeat + var_interaction + var_operator
    total = var_grr + var_part
    return {"parts": p, "operators": o, "repeats": r,
            "variance_repeat": var_repeat, "variance_interaction": var_interaction,
            "variance_operator": var_operator, "variance_part": var_part,
            "gauge_rr_sd": math.sqrt(var_grr), "percent_grr": 100 * math.sqrt(var_grr / total),
            "distinct_categories": math.floor(1.41 * math.sqrt(var_part / var_grr))
                                   if var_grr else None,
            "mean_squares": {"part": float(ms_part), "operator": float(ms_operator),
                             "interaction": float(ms_interaction), "repeat": float(ms_repeat)}}


def workstation_report(rows: list[Measurement], lower: float = 8, upper: float = 12) -> dict:
    result = {}
    for station in STATIONS:
        subset = [row for row in rows if row.workstation == station]
        errors = np.array([row.measured - row.true_value for row in subset])
        truth = np.array([row.true_value for row in subset])
        slope = linregress(truth, errors)
        result[station] = {"injected": STATIONS[station],
                           "anova": crossed_anova(subset),
                           "observed_bias": float(errors.mean()),
                           "bias_linearity_slope": float(slope.slope),
                           "capability": capability([row.measured for row in subset], lower, upper)}
    by_station_part: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_station_part[row.workstation, row.part].append(row.measured)
    parts = sorted({row.part for row in rows})
    differences = np.array([np.mean(by_station_part["new", part]) -
                            np.mean(by_station_part["old", part]) for part in parts])
    mean_diff = float(differences.mean())
    sd_diff = float(differences.std(ddof=1))
    result["agreement"] = {"new_minus_old_mean": mean_diff,
                           "bland_altman_lower": mean_diff - 1.96 * sd_diff,
                           "bland_altman_upper": mean_diff + 1.96 * sd_diff,
                           "parts_compared": len(parts)}
    return result


def spc_drift(seed: int = 7, points: int = 100, drift_start: int = 50) -> dict:
    """Detect a planted rising bias; limits are estimated from an in-control prefix."""
    if points < 40 or not 20 <= drift_start < points:
        raise ValueError("need an in-control prefix and post-change samples")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.03, points)
    drift = np.maximum(0, np.arange(points) - drift_start) * 0.002
    values = 10 + noise + drift
    baseline = values[:20]
    center = float(baseline.mean())
    sd = float(baseline.std(ddof=1))
    individuals = [i for i, value in enumerate(values) if abs(value - center) > 3 * sd]
    moving_ranges = np.abs(np.diff(values))
    mr_limit = 3.267 * float(moving_ranges[:19].mean())
    mr_flags = [i + 1 for i, value in enumerate(moving_ranges) if value > mr_limit]
    ewma = center
    ewma_flags = []
    lambda_ = 0.2
    for i, value in enumerate(values):
        ewma = lambda_ * value + (1 - lambda_) * ewma
        width = 3 * sd * math.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_) ** (2 * (i + 1))))
        if abs(ewma - center) > width:
            ewma_flags.append(i)
    plus = minus = 0.0
    cusum_flags = []
    for i, value in enumerate(values):
        plus = max(0.0, plus + value - center - 0.5 * sd)
        minus = max(0.0, minus + center - value - 0.5 * sd)
        if max(plus, minus) > 5 * sd:
            cusum_flags.append(i)
    return {"seed": seed, "points": points, "drift_start": drift_start,
            "baseline_sd": sd, "individuals_first_signal": individuals[0] if individuals else None,
            "moving_range_first_signal": mr_flags[0] if mr_flags else None,
            "ewma_first_signal": ewma_flags[0] if ewma_flags else None,
            "cusum_first_signal": cusum_flags[0] if cusum_flags else None}


def records(rows: list[Measurement]) -> list[dict]:
    return [asdict(row) for row in rows]
