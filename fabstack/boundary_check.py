"""Recheck the public-data and citation boundary without downloading anything.

1. Hash any locally present raw archive and compare it with the code constant,
   ``data/SOURCES.md`` and the hash recorded in the saved evidence JSON.
2. Check that every query ID used by insights, the router and the dashboard is
   documented in ``warehouse/QUERY_CATALOG.md`` and that cited rows resolve.
3. Render the dashboard in process and compare its metrics with the saved JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from fabstack import dashboard_evidence as ev

ROOT = ev.PROJECT_ROOT
RESULT = ROOT / "evidence/public_data_boundary.json"
SOURCES = ROOT / "data/SOURCES.md"
CATALOG = ROOT / "warehouse/QUERY_CATALOG.md"
# Module constants are read from source text so the check does not need pandas/XGBoost.
ARCHIVES = {
    "uci_secom": {"archive": "data/raw/secom.zip", "module": "fabstack/secom.py",
                  "evidence": "evidence/secom_baseline.json", "evidence_key": "archive_sha256"},
    "nasa_cmapss": {"archive": "data/raw/cmapss.zip", "module": "fabstack/cmapss.py",
                    "evidence": "evidence/cmapss_fd001.json", "evidence_key": "source_sha256"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_hash(source: str) -> str:
    match = re.search(r'^SHA256 = "([0-9a-f]{64})"', source, re.MULTILINE)
    if not match:
        raise ValueError("SHA256 constant not found")
    return match.group(1)


def documented_hashes(markdown: str) -> set[str]:
    return set(re.findall(r"\b[0-9a-f]{64}\b", markdown))


def catalog_ids(markdown: str) -> set[str]:
    return set(re.findall(r"^\| `([a-z0-9_]+_v\d+)` \|", markdown, re.MULTILINE))


def check_archives(root: Path = ROOT) -> dict:
    sources = documented_hashes((root / "data/SOURCES.md").read_text())
    report = {}
    for name, spec in ARCHIVES.items():
        expected = module_hash((root / spec["module"]).read_text())
        recorded = json.loads((root / spec["evidence"]).read_text())[spec["evidence_key"]]
        archive = root / spec["archive"]
        entry = {
            "archive": spec["archive"],
            "module_sha256": expected,
            "documented_in_sources_md": expected in sources,
            "evidence_json_sha256_matches": recorded == expected,
            "present_locally": archive.is_file(),
        }
        if archive.is_file():
            entry["bytes"] = archive.stat().st_size
            entry["local_sha256"] = sha256_file(archive)
            entry["local_matches_module"] = entry["local_sha256"] == expected
        entry["ok"] = (entry["documented_in_sources_md"] and entry["evidence_json_sha256_matches"]
                       and entry.get("local_matches_module", True))
        report[name] = entry
    return report


def check_citations(root: Path = ROOT) -> dict:
    documented = catalog_ids((root / "warehouse/QUERY_CATALOG.md").read_text())
    agent_ids = set(re.findall(r'template_id == "([a-z0-9_]+)"',
                               (root / "fabstack/agent.py").read_text()))
    dashboard_ids = set(re.findall(r"Query ID: ([a-z0-9_]+_v\d+)",
                                   (root / "dashboards/streamlit_app.py").read_text()))
    report = {"catalog_ids": sorted(documented), "agent_template_ids": sorted(agent_ids),
              "dashboard_query_ids": sorted(dashboard_ids)}
    database = ev.artifact_path(root, "database")
    insights_path = ev.artifact_path(root, "insights")
    agent_eval = root / "runs/utilities/agent_eval.json"
    if not (database.is_file() and insights_path.is_file() and agent_eval.is_file()):
        report["generated_artifacts_present"] = False
        report["ok"] = agent_ids <= documented and dashboard_ids <= documented
        return report
    from fabstack.agent import AnalyticsAgent
    from fabstack.warehouse import verify_insights

    insights = json.loads(insights_path.read_text())
    predictions = json.loads(agent_eval.read_text())["predictions"]
    cited = [citation for item in predictions for citation in item["citations"]]
    agent = AnalyticsAgent(database)
    with duckdb.connect(str(database), read_only=True) as connection:
        flagged = {row[0] for row in connection.execute(
            "select row_id from fact_meter_readings where flagged").fetchall()}
    replayed = sum(agent.answer(item["question"])["citations"] == item["citations"]
                   for item in predictions)
    report.update({
        "generated_artifacts_present": True,
        "insight_query_ids": sorted({item["query_id"] for item in insights}),
        "verify_insights": verify_insights(database),
        "router_citations": len(cited),
        "router_cited_rows_all_flagged": all(set(c["row_ids"]) <= flagged for c in cited),
        "router_saved_answers_replayed_identically": f"{replayed}/{len(predictions)}",
    })
    used = agent_ids | dashboard_ids | set(report["insight_query_ids"]) | {
        citation["query_id"] for citation in cited}
    report["undocumented_query_ids"] = sorted(used - documented)
    check = report["verify_insights"]
    report["ok"] = (not report["undocumented_query_ids"]
                    and check["insights"] == check["verified_source_row_totals"] == len(insights)
                    and report["router_cited_rows_all_flagged"]
                    and replayed == len(predictions))
    return report


def expected_metrics(root: Path = ROOT) -> dict:
    """Dashboard metric labels and values, formatted exactly as the app formats saved JSON."""
    expected = {}
    engine, _ = ev.load(root, "cmapss", ev.engine_view)
    utilities, _ = ev.load(root, "utilities", ev.utilities_view)
    msa, _ = ev.load(root, "msa", ev.msa_view)
    if utilities:
        expected.update({"Injected anomaly rows": str(utilities["injected_anomaly_rows"]),
                         "Precision": f"{utilities['precision']:.1%}",
                         "Recall": f"{utilities['recall']:.1%}"})
    if msa:
        expected.update({"Old %GRR": f"{msa['old_percent_grr']:.2f}%",
                         "New %GRR": f"{msa['new_percent_grr']:.2f}%",
                         "New − old bias": f"{msa['new_minus_old_mean']:.3f}"})
    if engine:
        expected.update({"Official test engines": str(engine["n"]),
                         "RMSE (cycles)": f"{engine['rmse']:.2f}",
                         "MAE (cycles)": f"{engine['mae']:.2f}"})
    return expected


def check_dashboard_numbers(root: Path = ROOT) -> dict:
    """Compare in-process dashboard metrics with values formatted from the saved JSON."""
    from streamlit.testing.v1 import AppTest

    expected = expected_metrics(root)
    previous = os.environ.get(ev.ROOT_ENV)
    os.environ[ev.ROOT_ENV] = str(root)
    try:
        app = AppTest.from_file(str(ev.PROJECT_ROOT / "dashboards/streamlit_app.py"), default_timeout=30).run()
    finally:
        if previous is None:
            os.environ.pop(ev.ROOT_ENV, None)
        else:
            os.environ[ev.ROOT_ENV] = previous
    shown = {metric.label: metric.value for metric in app.metric}
    return {"method": "Streamlit AppTest, in process (not a browser)",
            "exceptions": [error.message for error in app.exception],
            "expected_from_json": expected, "shown": shown,
            "ok": not app.exception and shown == expected}


def run(root: Path = ROOT) -> dict:
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "network": "none; archives are hashed only if already present in ignored data/raw/",
        "archives": check_archives(root),
        "citations": check_citations(root),
        "dashboard_numbers": check_dashboard_numbers(root),
    }
    report["ok"] = (all(entry["ok"] for entry in report["archives"].values())
                    and report["citations"]["ok"] and report["dashboard_numbers"]["ok"])
    return report


def main() -> int:
    report = run()
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": report["ok"], "evidence": str(RESULT.relative_to(ROOT))}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
