"""Five small, reproducible manufacturing analytics modules on synthetic data."""
import argparse
import json
import math
import random
from statistics import mean, median, stdev


def yield_screen(seed=7, n=300):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        temperature = rng.gauss(40, 4)
        vibration = rng.gauss(2, .5)
        fail = int(temperature > 45 or vibration > 2.8)
        rows.append((temperature, vibration, fail))
    threshold = 45
    predicted = [int(row[0] > threshold) for row in rows]
    actual = [row[2] for row in rows]
    tp = sum(p and y for p, y in zip(predicted, actual))
    return {"n": n, "failures": sum(actual), "temperature_rule_precision": tp / max(1, sum(predicted)), "temperature_rule_recall": tp / max(1, sum(actual))}


def rul_baseline(cycles, cap=125):
    if not cycles or min(cycles) < 0:
        raise ValueError("nonnegative cycles required")
    end = max(cycles)
    return [min(cap, end-c) for c in cycles]


def utility_anomalies(values, window=24, threshold=4):
    flags = []
    for i, value in enumerate(values):
        history = values[max(0, i-window):i]
        if len(history) < 5:
            continue
        baseline = median(history)
        mad = median(abs(x-baseline) for x in history) or 1e-9
        if abs(value-baseline) / (1.4826*mad) > threshold:
            flags.append(i)
    return flags


def simulate_train(max_train_size, seed=7, lots=100):
    if max_train_size < 1:
        raise ValueError("train size must be positive")
    rng = random.Random(seed)
    queue = [rng.choice(("A", "B", "C")) for _ in range(lots)]
    setup_time = 5
    process_time = 2
    time = setups = 0
    current = None
    streak = 0
    for recipe in queue:
        if recipe != current or streak >= max_train_size:
            time += setup_time
            setups += 1
            current = recipe
            streak = 0
        time += process_time
        streak += 1
    return {"lots": lots, "setups": setups, "elapsed_minutes": time, "throughput_lots_per_hour": 60*lots/time}


def capability(values, lower, upper):
    sigma = stdev(values)
    if sigma == 0:
        raise ValueError("nonzero variation required")
    mu = mean(values)
    return {"cp": (upper-lower)/(6*sigma), "cpk": min(upper-mu, mu-lower)/(3*sigma)}


def gauge_rr(parts=10, operators=3, repeats=2, seed=7):
    rng = random.Random(seed)
    true = [rng.gauss(10, 1) for _ in range(parts)]
    bias = [rng.gauss(0, .05) for _ in range(operators)]
    observations = {(p, o): [true[p]+bias[o]+rng.gauss(0, .03) for _ in range(repeats)] for p in range(parts) for o in range(operators)}
    repeatability = mean(stdev(v) for v in observations.values())
    operator_means = [mean(x for (p, op), vals in observations.items() if op == o for x in vals) for o in range(operators)]
    return {"repeatability_sd_mean": repeatability, "operator_mean_range": max(operator_means)-min(operator_means)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)
    utilities = [100 + rng.gauss(0, 2) for _ in range(80)]
    utilities[60] += 30
    print(json.dumps({"yield": yield_screen(args.seed), "rul_last": rul_baseline(list(range(10)))[-1], "utility_flags": utility_anomalies(utilities), "train_sim": simulate_train(4, args.seed), "capability": capability([rng.gauss(10, .5) for _ in range(100)], 8, 12), "gauge_rr": gauge_rr(seed=args.seed)}, indent=2))


if __name__ == "__main__":
    main()
