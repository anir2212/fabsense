# FabSense

Synthetic manufacturing analytics and simulation.

A set of manufacturing analytics exercises: photo-bay simulation, gauge repeatability and reproducibility, facility meter anomalies, and a bounded SQL question router over DuckDB.

![Simulated photo-bay throughput tradeoff](figures/photo_tradeoff.png)

## Why this project

Manufacturing improvements are easy to overstate when process variability, measurement error, and utility signals are mixed together. This project keeps those questions separate and makes each conclusion traceable to a seeded calculation.

## Run locally

You need Python 3.12.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make all
make test
```

`make dashboard` opens the Streamlit view. The [query catalog](warehouse/QUERY_CATALOG.md) lists supported questions, while [results](RESULTS.md) records the seeded runs.

## Repository map

`fabstack/` contains the analyses and question router; `warehouse/` holds the SQL contract; `dashboards/` provides the viewer.

## Scope

All inputs are generated. The simulation and meter flags have not been calibrated against a real fab.
