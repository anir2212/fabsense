"""Build seeded meter data, evaluate injected anomalies and materialize cited insights."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fabstack.utilities import detect, generate, records
from fabstack.warehouse import build, verify_insights


def run(directory: Path = Path("runs/utilities"), seed: int = 7) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    original = generate(seed=seed)
    holdout, metrics = detect(original)
    combined = [row for row in original if row.hour_index < 240] + holdout
    with (directory / "meter_readings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records(combined)[0]))
        writer.writeheader()
        writer.writerows(records(combined))
    database = directory / "fabsense.duckdb"
    insights = build(combined, database)
    verification = verify_insights(database)
    report = {"seed": seed, "anomaly": metrics, "insight_verification": verification,
              "insight_count": len(insights), "database": str(database)}
    (directory / "insights.json").write_text(json.dumps(insights, indent=2) + "\n")
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
