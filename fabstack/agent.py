"""Allow-listed DuckDB tools and a deterministic cited question router."""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from fabstack.utilities import METRICS


class AnalyticsAgent:
    def __init__(self, database: Path):
        self.database = database

    def run_sql(self, template_id: str, params: dict) -> dict:
        """Execute only named parameterized templates, never caller-supplied SQL."""
        with duckdb.connect(str(self.database), read_only=True) as connection:
            if template_id == "top_building_excess_v1":
                metric = params.get("metric")
                if metric not in METRICS:
                    raise ValueError("known metric required")
                row = connection.execute(
                    "select building_id, sum(greatest(observed-predicted,0)) as excess, "
                    "list(row_id) as source_ids from fact_meter_readings "
                    "where flagged and metric=? group by building_id "
                    "order by excess desc, building_id limit 1", [metric]
                ).fetchone()
                return {"query_id": template_id, "metric": metric,
                        "building": row[0] if row else None,
                        "excess_units": float(row[1]) if row else 0.0,
                        "row_ids": sorted(row[2]) if row else []}
            if template_id == "building_meter_excess_v1":
                building, metric = params.get("building"), params.get("metric")
                if not re.fullmatch(r"B\d{2}", str(building)) or metric not in METRICS:
                    raise ValueError("known building and metric required")
                row = connection.execute(
                    "select count(*), coalesce(sum(greatest(observed-predicted,0)),0), "
                    "list(row_id) from fact_meter_readings "
                    "where flagged and building_id=? and metric=?", [building, metric]
                ).fetchone()
                return {"query_id": template_id, "building": building, "metric": metric,
                        "flagged_hours": row[0], "excess_units": float(row[1]),
                        "row_ids": sorted(row[2] or [])}
            raise ValueError("template_id is not allow-listed")

    def get_insight(self, insight_id: str) -> dict:
        if not re.fullmatch(r"I-\d{3}", insight_id):
            raise ValueError("invalid insight identifier")
        with duckdb.connect(str(self.database), read_only=True) as connection:
            row = connection.execute(
                "select insight_id, query_id, row_ids_json, sentence from fact_insights "
                "where insight_id=?", [insight_id]
            ).fetchone()
        if row is None:
            raise KeyError(insight_id)
        return {"insight_id": row[0], "query_id": row[1],
                "row_ids": json.loads(row[2]), "sentence": row[3]}

    @staticmethod
    def explain_metric(name: str) -> str:
        definitions = {
            "excess": "Positive observed use minus the weather-adjusted baseline on flagged hours.",
            "precision": "True flagged anomalies divided by all flagged rows in the injected holdout.",
            "recall": "True flagged anomalies divided by all injected anomaly rows in the holdout.",
        }
        if name not in definitions:
            raise ValueError("unknown metric")
        return definitions[name]

    @staticmethod
    def search_docs(query: str, root: Path = Path(".")) -> list[dict]:
        words = set(re.findall(r"[a-z]{4,}", query.lower()))
        matches = []
        for path in (root / "README.md", root / "RESULTS.md"):
            if not path.exists():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if words and words.intersection(re.findall(r"[a-z]{4,}", line.lower())):
                    matches.append({"file": str(path), "line": number, "text": line[:200]})
                    if len(matches) >= 5:
                        return matches
        return matches

    def answer(self, question: str) -> dict:
        lower = question.lower()
        metric = next((name for name in METRICS if name.replace("_", " ") in lower), None)
        if metric is None:
            return {"answer": "I cannot identify a supported meter in the question.",
                    "citations": []}
        building = re.search(r"\bB\d{2}\b", question, re.IGNORECASE)
        if building:
            result = self.run_sql("building_meter_excess_v1",
                                  {"building": building.group().upper(), "metric": metric})
            sentence = (f"{result['building']} {metric.replace('_', ' ')} had "
                        f"{result['flagged_hours']} flagged hours and "
                        f"{result['excess_units']:.1f} excess units.")
        elif any(word in lower for word in ("top", "largest", "highest", "most")):
            result = self.run_sql("top_building_excess_v1", {"metric": metric})
            sentence = (f"{result['building']} had the largest flagged {metric.replace('_', ' ')} "
                        f"excess: {result['excess_units']:.1f} units.")
        else:
            return {"answer": "Specify a building or ask for the top building.",
                    "citations": []}
        return {"answer": sentence, "citations": [
            {"query_id": result["query_id"], "row_ids": result["row_ids"]}]}
