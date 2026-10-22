"""Load and validate the saved evidence that each dashboard tab displays.

Every loader either returns the exact values the tab shows or raises
``EvidenceProblem`` naming the file and the command that rebuilds it. The
dashboard never substitutes defaults for missing or malformed evidence.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from fabstack.utilities import METRICS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_ENV = "FABSENSE_ARTIFACT_ROOT"


@dataclass(frozen=True)
class Artifact:
    path: str
    command: str


ARTIFACTS = {
    "photo": Artifact("runs/photo_grid.json", "make photo"),
    "photo_figure": Artifact("figures/photo_tradeoff.png", "make photo msa figures"),
    "utilities": Artifact("runs/utilities/results.json", "make utilities"),
    "insights": Artifact("runs/utilities/insights.json", "make utilities"),
    "database": Artifact("runs/utilities/fabsense.duckdb", "make utilities"),
    "msa": Artifact("runs/msa/results.json", "make msa"),
    "msa_figure": Artifact("figures/msa_agreement.png", "make photo msa figures"),
    "cmapss": Artifact("evidence/cmapss_fd001.json", "make cmapss"),
}


class EvidenceProblem(Exception):
    def __init__(self, key: str, kind: str, detail: str = ""):
        self.key, self.kind, self.detail = key, kind, detail
        self.artifact = ARTIFACTS[key]
        super().__init__(self.message)

    @property
    def message(self) -> str:
        reason = f" ({self.detail})" if self.detail else ""
        return (f"{self.kind.capitalize()} evidence file `{self.artifact.path}`{reason}. "
                f"No values are shown from it. Regenerate it with `{self.artifact.command}`.")


def artifact_root() -> Path:
    """Project root by default; tests point this at a temporary artifact tree."""
    return Path(os.environ.get(ROOT_ENV) or PROJECT_ROOT).resolve()


def artifact_path(root: Path, key: str) -> Path:
    return root / ARTIFACTS[key].path


def require_file(root: Path, key: str) -> Path:
    path = artifact_path(root, key)
    if not path.exists():
        raise EvidenceProblem(key, "missing")
    if not path.is_file():
        raise EvidenceProblem(key, "malformed", "not a regular file")
    return path


def read_json(root: Path, key: str):
    path = require_file(root, key)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceProblem(key, "malformed", f"not valid JSON: {type(error).__name__}") from None


def load(root: Path, key: str, view):
    """Return ``(values, None)`` or ``(None, EvidenceProblem)`` for one evidence file."""
    try:
        data = read_json(root, key)
        try:
            return view(data), None
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            raise EvidenceProblem(key, "malformed", f"unexpected structure: {error}") from None
    except EvidenceProblem as problem:
        return None, problem


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} is not a finite number")
    return float(value)


def count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} is not a non-negative integer")
    return value


def fraction(value, name: str) -> float:
    value = number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} is outside 0-1")
    return value


def text(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is not text")
    return value


def mapping(value, name: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} is not a non-empty object")
    return value


def sequence(value, name: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} is not a non-empty list")
    return value


PHOTO_COLUMNS = ("throughput_lots_per_day", "p90_cycle_minutes", "setup_time_share", "backlog")


def photo_view(data: dict) -> dict:
    scenarios = []
    for row in sequence(data["scenarios"], "scenarios"):
        summary = mapping(row["summary"], "summary")
        scenarios.append({
            "mix": text(row["mix"], "mix"),
            "arrival_rate_per_minute": number(row["arrival_rate_per_minute"], "arrival rate"),
            "max_train_size": count(row["max_train_size"], "max_train_size"),
            **{name: number(summary[name]["mean"], name) for name in PHOTO_COLUMNS},
        })
    recommendations = {
        (text(item["mix"], "mix"), number(item["arrival_rate_per_minute"], "arrival rate")):
            count(item["selected"]["train_size"], "train_size")
        for item in sequence(data["recommendations"], "recommendations")
    }
    return {
        "mixes": sorted({row["mix"] for row in scenarios}),
        "rates": sorted({row["arrival_rate_per_minute"] for row in scenarios}),
        "scenarios": scenarios,
        "recommendations": recommendations,
    }


def utilities_view(data: dict) -> dict:
    anomaly = mapping(data["anomaly"], "anomaly")
    return {
        "injected_anomaly_rows": count(anomaly["injected_anomaly_rows"], "injected_anomaly_rows"),
        "precision": fraction(anomaly["precision"], "precision"),
        "recall": fraction(anomaly["recall"], "recall"),
    }


def insights_view(data: list) -> list[dict]:
    insights = []
    for item in sequence(data, "insights"):
        row_ids = sequence(item["row_ids"], "row_ids")
        insights.append({
            "building": text(item["building"], "building"),
            "metric": text(item["metric"], "metric"),
            "query_id": text(item["query_id"], "query_id"),
            "row_ids": [count(row_id, "row_id") for row_id in row_ids],
            "sentence": text(item["sentence"], "sentence"),
        })
    return insights


def msa_view(data: dict) -> dict:
    stations = mapping(data["workstations"], "workstations")
    return {
        "old_percent_grr": number(stations["old"]["anova"]["percent_grr"], "old percent_grr"),
        "new_percent_grr": number(stations["new"]["anova"]["percent_grr"], "new percent_grr"),
        "new_minus_old_mean": number(stations["agreement"]["new_minus_old_mean"],
                                     "new_minus_old_mean"),
        "spc": mapping(data["spc"], "spc"),
    }


def engine_view(data: dict) -> dict:
    test = mapping(data["official_test"], "official_test")
    interval = sequence(data["official_test_ci_95"]["rmse_95"], "rmse_95")
    if len(interval) != 2:
        raise ValueError("rmse_95 must have two bounds")
    return {
        "n": count(test["n"], "official_test.n"),
        "rmse": number(test["rmse"], "official_test.rmse"),
        "mae": number(test["mae"], "official_test.mae"),
        "interval": [number(bound, "rmse_95") for bound in interval],
        "selected_model": text(data["selected_model"], "selected_model"),
        "validation_endpoint_count": count(data["validation_endpoint_count"],
                                           "validation_endpoint_count"),
        "validation": [
            {"candidate": name, "validation RMSE (cycles)": number(values["rmse"], name)}
            for name, values in mapping(data["validation"], "validation").items()
        ],
        "slices": [
            {"slice": name, "engines": count(values["n"], name),
             "RMSE (cycles)": number(values["rmse"], name)}
            for name, values in mapping(data["official_test_rul_slices"], "slices").items()
        ],
    }


def engine_tables(engine: dict) -> dict:
    """FD001 table rows formatted to two decimals, as shown in the dashboard and RESULTS.md."""
    return {
        "validation": [{"candidate": row["candidate"],
                        "validation RMSE (cycles)": f"{row['validation RMSE (cycles)']:.2f}"}
                       for row in engine["validation"]],
        "slices": [{"slice": row["slice"], "engines": str(row["engines"]),
                    "RMSE (cycles)": f"{row['RMSE (cycles)']:.2f}"} for row in engine["slices"]],
    }


def markdown_table(rows: list[dict]) -> str:
    """Static Markdown table without a positional index column.

    Streamlit 1.50's ``st.table`` ignores ``Styler.hide(axis="index")`` and would show
    0, 1, 2 row numbers; a Markdown table renders as a plain HTML table instead.
    """
    if not rows:
        raise ValueError("no rows to render")

    def cell(value) -> str:
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("_", "\\_")

    header = list(rows[0])
    lines = ["| " + " | ".join(cell(name) for name in header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(row[name]) for name in header) + " |" for row in rows]
    return "\n".join(lines)


def database_view(root: Path) -> dict:
    """Open the generated DuckDB file read-only and list the selectable entities."""
    path = require_file(root, "database")
    try:
        with duckdb.connect(str(path), read_only=True) as connection:
            buildings = [row[0] for row in connection.execute(
                "select distinct building_id from fact_meter_readings order by building_id"
            ).fetchall()]
            present = {row[0] for row in connection.execute(
                "select distinct metric from fact_meter_readings").fetchall()}
    except duckdb.Error as error:
        raise EvidenceProblem("database", "malformed",
                              f"DuckDB could not read it: {type(error).__name__}") from None
    # Keep the generator's meter order so electricity stays the default selection.
    metrics = [name for name in METRICS if name in present]
    if not buildings or not metrics:
        raise EvidenceProblem("database", "malformed", "fact_meter_readings is empty")
    return {"path": path, "buildings": buildings, "metrics": metrics}


def load_database(root: Path):
    try:
        return database_view(root), None
    except EvidenceProblem as problem:
        return None, problem


def figure(root: Path, key: str):
    try:
        return require_file(root, key), None
    except EvidenceProblem as problem:
        return None, problem
