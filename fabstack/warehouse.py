"""DuckDB star-schema load and source-row-cited utility insight generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import duckdb

from fabstack.utilities import Reading


def build(readings: list[Reading], database: Path,
          schema: Path = Path("warehouse/star_schema.sql")) -> list[dict]:
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()  # This path is a generated, project-owned artifact.
    connection = duckdb.connect(str(database))
    try:
        connection.execute(schema.read_text())
        buildings = sorted({row.building for row in readings})
        connection.executemany("insert into dim_building values (?, ?)",
                               [(building, "Synthetic facility " + building) for building in buildings])
        by_hour = {row.hour_index: (row.hour_index, row.hour_of_day, row.weekday)
                   for row in readings}
        connection.executemany("insert into dim_time values (?, ?, ?)", list(by_hour.values()))
        connection.executemany("insert into fact_meter_readings values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(row.row_id, row.building, row.metric, row.hour_index, row.temperature_c,
              row.observed, row.predicted, row.flagged, row.anomaly_type) for row in readings])
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in readings:
            if row.flagged:
                groups[row.building, row.metric].append(row.row_id)
        insights = []
        for index, ((building, metric), row_ids) in enumerate(sorted(groups.items()), start=1):
            placeholders = ",".join("?" for _ in row_ids)
            excess = connection.execute(
                "select coalesce(sum(greatest(observed - predicted, 0)), 0) "
                f"from fact_meter_readings where row_id in ({placeholders})", row_ids
            ).fetchone()[0]
            unit = "m³" if metric == "water" else "kWh"
            sentence = (f"{building} {metric.replace('_', ' ')} had {len(row_ids)} flagged "
                        f"hours and {excess:.1f} {unit} above the weather-adjusted baseline.")
            insight = {"insight_id": f"I-{index:03d}", "building": building,
                       "metric": metric, "query_id": "flagged_excess_by_rows_v1",
                       "row_ids": row_ids, "excess_units": float(excess), "sentence": sentence}
            insights.append(insight)
            connection.execute("insert into fact_insights values (?, ?, ?, ?, ?, ?, ?)",
                               [insight["insight_id"], building, metric, insight["query_id"],
                                json.dumps(row_ids), float(excess), sentence])
        connection.commit()
        return insights
    finally:
        connection.close()


def verify_insights(database: Path) -> dict:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute("select insight_id, row_ids_json, excess_units from fact_insights").fetchall()
        matched = 0
        for insight_id, encoded, recorded in rows:
            row_ids = json.loads(encoded)
            placeholders = ",".join("?" for _ in row_ids)
            count, excess = connection.execute(
                "select count(*), coalesce(sum(greatest(observed - predicted, 0)), 0) "
                f"from fact_meter_readings where row_id in ({placeholders})", row_ids
            ).fetchone()
            if count == len(row_ids) and abs(excess - recorded) < 1e-9:
                matched += 1
        return {"insights": len(rows), "verified_source_row_totals": matched}
    finally:
        connection.close()
