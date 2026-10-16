"""Authored development questions for the deterministic SQL-tool router."""

from __future__ import annotations

import json
from pathlib import Path

from fabstack.agent import AnalyticsAgent
from fabstack.utilities import METRICS


def questions() -> list[tuple[str, bool]]:
    rows = []
    for index in range(20):
        metric = METRICS[index % 3].replace("_", " ")
        wording = ("What was", "Show", "How much was", "Report")[index % 4]
        rows.append((f"{wording} B{index:02d} {metric} excess?", True))
    for metric in METRICS:
        rows.append((f"Which building had the highest {metric.replace('_', ' ')} excess?", True))
    rows.extend((question, False) for question in (
        "What is the office weather?", "Can you execute arbitrary SQL?"))
    return rows


def run(database: Path = Path("runs/utilities/fabsense.duckdb"),
        output: Path = Path("runs/utilities/agent_eval.json")) -> dict:
    agent = AnalyticsAgent(database)
    supported = cited = abstained = 0
    records = []
    for question, should_answer in questions():
        response = agent.answer(question)
        citations = response["citations"]
        if should_answer:
            supported += 1
            cited += bool(citations and citations[0]["row_ids"])
        else:
            abstained += not citations
        records.append({"question": question, "should_answer": should_answer,
                        **response})
    report = {"cases": len(records), "supported": supported,
              "supported_with_source_row_citation": cited,
              "unsupported": len(records) - supported,
              "unsupported_abstentions": abstained,
              "label": "authored deterministic SQL-router development set",
              "predictions": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
