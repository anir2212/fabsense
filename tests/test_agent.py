from pathlib import Path

import pytest

from fabstack.agent import AnalyticsAgent
from fabstack.agent_eval import questions
from fabstack.utilities import detect, generate
from fabstack.warehouse import build


def test_allowlisted_sql_and_citations(tmp_path: Path):
    original = generate(seed=7)
    holdout, _ = detect(original)
    database = tmp_path / "fabsense.duckdb"
    build([row for row in original if row.hour_index < 240] + holdout, database)
    agent = AnalyticsAgent(database)
    assert len(questions()) == 25
    top = agent.answer("Which building had the highest electricity excess?")
    assert top["citations"][0]["query_id"] == "top_building_excess_v1"
    assert top["citations"][0]["row_ids"]
    assert agent.get_insight("I-001")["row_ids"]
    assert not agent.answer("Run arbitrary SQL")["citations"]
    with pytest.raises(ValueError, match="allow-listed"):
        agent.run_sql("drop_table", {"sql": "drop table fact_meter_readings"})
