# FabSense

Manufacturing analytics with source-aware evidence.

The local workbench covers photo-bay simulation, gauge repeatability and reproducibility, facility meter anomalies, and a bounded SQL question router over DuckDB. Those modules use seeded data. A separate opt-in baseline uses the attributed UCI SECOM dataset to test yield-failure ranking on recorded sensor rows.

![Simulated photo-bay throughput tradeoff](figures/photo_tradeoff.png)

## Why this project

Manufacturing improvements are easy to overstate when process variability, measurement error, and utility signals are mixed together. This project keeps those questions separate and makes each conclusion traceable to a seeded calculation.

SECOM adds a different question: how well does a classifier trained on earlier recorded test points rank later in-house failures when only anonymous sensor columns are available? Its result is useful partly because it shows how model selection and a strict alert threshold can fail on a later slice.

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

The SECOM run is independent of `make all`:

```sh
.venv/bin/python -m pip install -r requirements-secom.txt
make secom
```

That command downloads UCI's CC BY 4.0 archive into ignored `data/raw/`, verifies its pinned SHA-256, and writes [a compact evidence file](evidence/secom_baseline.json). It uses a timestamp-ordered 1,253/314 development/test split with preprocessing inside five development folds. XGBoost was selected by development average precision, but reached only **0.0890** AP on the later test slice; the test positive-rate baseline was **0.0541**. Logistic regression reached **0.2250** AP on that same slice, but selecting it after reading the test would bias the result. All three models raised zero alerts under their development-selected 50% precision thresholds. [RESULTS.md](RESULTS.md) gives fold variation, review-capacity results and limits.

## Repository map

`fabstack/` contains the analyses, SECOM baseline and question router; `warehouse/` holds the SQL contract; `dashboards/` provides the viewer. [data/SOURCES.md](data/SOURCES.md) attributes UCI and separates that dataset from generated inputs.

## Scope

The photo, MSA and facilities inputs are generated, and those modules are not calibrated against a real fab. SECOM is a single historical public dataset with anonymous sensors and 17 failures in its later 314-row test slice. It does not establish a causal sensor explanation, a prospective yield screen, or an operating threshold. Wafer-map, equipment-RUL and public building benchmarks remain open.
