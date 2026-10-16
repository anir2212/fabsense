from fabstack.utilities import detect, generate
from fabstack.warehouse import build, verify_insights


def test_anomaly_labels_are_held_out_and_detection_is_measured():
    rows = generate(seed=7)
    assert len(rows) == 20 * 3 * 14 * 24
    assert sum(row.anomaly_type is not None for row in rows) == 30
    assert all(row.anomaly_type is None for row in rows if row.hour_index < 240)
    _, report = detect(rows)
    assert report["injected_anomaly_rows"] == 30
    assert report["true_positive"] > 0
    assert report["precision"] <= 1 and report["recall"] <= 1


def test_every_insight_resolves_to_rows_and_sql_sum(tmp_path):
    original = generate(seed=7)
    holdout, _ = detect(original)
    database = tmp_path / "facilities.duckdb"
    insights = build([row for row in original if row.hour_index < 240] + holdout, database)
    checked = verify_insights(database)
    assert checked["verified_source_row_totals"] == len(insights)
    assert all(insight["query_id"] == "flagged_excess_by_rows_v1" for insight in insights)
