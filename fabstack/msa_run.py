"""Generate crossed measurements and save both raw rows and analyzed outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fabstack.msa import generate, records, spc_drift, workstation_report


def run(directory: Path = Path("runs/msa"), seed: int = 7) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    observations = generate(seed=seed)
    with (directory / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records(observations)[0]))
        writer.writeheader()
        writer.writerows(records(observations))
    report = {"seed": seed, "observations": len(observations),
              "workstations": workstation_report(observations),
              "spc": spc_drift(seed=seed)}
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
