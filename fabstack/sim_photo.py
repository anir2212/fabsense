"""Stylised SimPy photo bay with scanner trains and shared reticle resources."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import simpy


@dataclass(frozen=True)
class Scenario:
    max_train_size: int
    arrival_rate_per_minute: float
    recipe_mix: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    scanners: int = 2
    horizon_minutes: int = 7 * 24 * 60
    setup_minutes: int = 12
    process_minutes: int = 20
    due_after_minutes: int = 180


@dataclass(frozen=True)
class Lot:
    number: int
    recipe: str
    arrived: float
    due: float


def simulate(config: Scenario, seed: int = 7) -> dict:
    if config.max_train_size < 1 or config.arrival_rate_per_minute <= 0 or config.scanners < 1:
        raise ValueError("positive train size, arrival rate and scanner count required")
    if len(config.recipe_mix) != 3 or any(weight < 0 for weight in config.recipe_mix) or abs(sum(config.recipe_mix) - 1) > 1e-8:
        raise ValueError("three recipe weights must sum to one")
    rng = random.Random(seed)
    env = simpy.Environment()
    recipes = ("A", "B", "C")
    reticles = {recipe: simpy.Resource(env, capacity=1) for recipe in recipes}
    waiting: list[Lot] = []
    completed: list[tuple[Lot, float]] = []
    metrics = {"arrived": 0, "setups": 0, "setup_minutes": 0.0,
               "process_minutes": 0.0, "reticle_wait_minutes": 0.0}

    def arrivals():
        number = 0
        while True:
            yield env.timeout(rng.expovariate(config.arrival_rate_per_minute))
            if env.now >= config.horizon_minutes:
                return
            recipe = rng.choices(recipes, weights=config.recipe_mix)[0]
            waiting.append(Lot(number, recipe, env.now,
                               env.now + config.due_after_minutes))
            metrics["arrived"] += 1
            number += 1

    def scanner():
        while True:
            if not waiting:
                yield env.timeout(1)
                continue
            # Earliest due lot chooses the next reticle/train.
            first = min(waiting, key=lambda lot: (lot.due, lot.number))
            waiting.remove(first)
            recipe = first.recipe
            request_started = env.now
            with reticles[recipe].request() as request:
                yield request
                metrics["reticle_wait_minutes"] += env.now - request_started
                yield env.timeout(config.setup_minutes)
                metrics["setups"] += 1
                metrics["setup_minutes"] += config.setup_minutes
                lot = first
                for index in range(config.max_train_size):
                    yield env.timeout(config.process_minutes)
                    metrics["process_minutes"] += config.process_minutes
                    completed.append((lot, env.now))
                    if index + 1 == config.max_train_size:
                        break
                    matching = [queued for queued in waiting if queued.recipe == recipe]
                    if not matching:
                        break
                    lot = min(matching, key=lambda queued: (queued.due, queued.number))
                    waiting.remove(lot)

    env.process(arrivals())
    for _ in range(config.scanners):
        env.process(scanner())
    env.run(until=config.horizon_minutes)
    cycles = np.array([finish - lot.arrived for lot, finish in completed])
    on_time = sum(finish <= lot.due for lot, finish in completed)
    capacity = config.scanners * config.horizon_minutes
    return {"seed": seed, "scenario": asdict(config), "arrived": metrics["arrived"],
            "completed": len(completed), "backlog": metrics["arrived"] - len(completed),
            "throughput_lots_per_day": len(completed) / (config.horizon_minutes / 1440),
            "mean_cycle_minutes": float(cycles.mean()) if len(cycles) else None,
            "p90_cycle_minutes": float(np.percentile(cycles, 90)) if len(cycles) else None,
            "scanner_utilisation": (metrics["setup_minutes"] + metrics["process_minutes"]) / capacity,
            "setup_time_share": metrics["setup_minutes"] /
                                max(1, metrics["setup_minutes"] + metrics["process_minutes"]),
            "on_time_fraction": on_time / len(completed) if completed else None,
            "mean_reticle_wait_minutes_per_arrival": metrics["reticle_wait_minutes"] /
                                                      max(1, metrics["arrived"]),
            "setups": metrics["setups"]}
