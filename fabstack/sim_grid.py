"""Twenty-replication train-size grid, intervals and a fitted scenario surrogate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from fabstack.sim_photo import Scenario, simulate

TRAIN_SIZES = (1, 2, 4, 8, 12, 16, 24)
LOADS = (0.05, 0.065, 0.08)
MIXES = {"balanced": (1 / 3, 1 / 3, 1 / 3), "skewed": (0.65, 0.25, 0.10)}
KEYS = ("throughput_lots_per_day", "mean_cycle_minutes", "p90_cycle_minutes",
        "scanner_utilisation", "setup_time_share", "on_time_fraction",
        "mean_reticle_wait_minutes_per_arrival", "backlog")


def interval(values: list[float]) -> list[float]:
    numbers = np.asarray(values, dtype=float)
    center = float(numbers.mean())
    if len(numbers) < 2:
        return [center, center]
    margin = float(t.ppf(0.975, len(numbers) - 1) * numbers.std(ddof=1) / math.sqrt(len(numbers)))
    return [center - margin, center + margin]


def features(size: int, rate: float, skewed: bool) -> list[float]:
    return [math.log2(size), rate, int(skewed)]


def run(repetitions: int = 20, output: Path = Path("runs/photo_grid.json")) -> dict:
    if repetitions < 2:
        raise ValueError("at least two replications required for intervals")
    rows, training_x, training_y, holdout_x, holdout_y = [], [], {key: [] for key in (KEYS[0], KEYS[2])}, [], {key: [] for key in (KEYS[0], KEYS[2])}
    for mix_name, mix in MIXES.items():
        for rate in LOADS:
            for size in TRAIN_SIZES:
                scenario = Scenario(size, rate, mix)
                results = [simulate(scenario, seed=100 + repeat) for repeat in range(repetitions)]
                summary = {key: {"mean": float(np.mean([item[key] for item in results])),
                                 "ci95": interval([item[key] for item in results])}
                           for key in KEYS}
                rows.append({"mix": mix_name, "arrival_rate_per_minute": rate,
                             "max_train_size": size, "replications": repetitions,
                             "summary": summary})
                for repeat, result in enumerate(results):
                    target_x = holdout_x if repeat % 5 == 0 else training_x
                    target_y = holdout_y if repeat % 5 == 0 else training_y
                    target_x.append(features(size, rate, mix_name == "skewed"))
                    for key in training_y:
                        target_y[key].append(result[key])
    surrogate = {}
    models = {}
    for key in training_y:
        model = GradientBoostingRegressor(random_state=7, n_estimators=100, max_depth=3)
        model.fit(training_x, training_y[key])
        predicted = model.predict(holdout_x)
        surrogate[key + "_holdout_seed_mae"] = float(mean_absolute_error(holdout_y[key], predicted))
        models[key] = model
    recommendations = []
    for mix_name in MIXES:
        for rate in LOADS:
            options = []
            for size in TRAIN_SIZES:
                x = [features(size, rate, mix_name == "skewed")]
                throughput = float(models[KEYS[0]].predict(x)[0])
                p90 = float(models[KEYS[2]].predict(x)[0])
                # A documented example utility, not a fab-specific economic objective.
                utility = throughput - 0.03 * max(0, p90 - 180)
                options.append({"train_size": size, "predicted_throughput": throughput,
                                "predicted_p90_cycle": p90, "example_utility": utility})
            recommendations.append({"mix": mix_name, "arrival_rate_per_minute": rate,
                                    "selected": max(options, key=lambda row: row["example_utility"]),
                                    "options": options})
    report = {"model": "stylised SimPy photo bay; not fab-calibrated",
              "scenario_count": len(rows), "total_replications": len(rows) * repetitions,
              "surrogate_holdout": surrogate,
              "example_utility": "throughput/day - 0.03*max(0,p90_cycle_minutes-180)",
              "recommendations": recommendations, "scenarios": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("runs/photo_grid.json"))
    args = parser.parse_args()
    report = run(args.repetitions, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ("scenarios", "recommendations")}, indent=2))
