"""Dashboard evidence loaders must name bad files instead of inventing values."""

import json
import math
import shutil
from pathlib import Path

import duckdb
import pytest

from fabstack import dashboard_evidence as ev

ROOT = Path(__file__).resolve().parents[1]


def write(root: Path, key: str, payload) -> Path:
    path = ev.artifact_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_missing_file_names_path_and_command(tmp_path):
    values, problem = ev.load(tmp_path, "cmapss", ev.engine_view)
    assert values is None
    assert problem.kind == "missing"
    assert "`evidence/cmapss_fd001.json`" in problem.message
    assert "`make cmapss`" in problem.message


@pytest.mark.parametrize("payload, detail", [
    ("{not json", "not valid JSON"),
    ({"official_test": {}}, "unexpected structure"),
    ([1, 2, 3], "unexpected structure"),
])
def test_malformed_engine_evidence_is_rejected(tmp_path, payload, detail):
    write(tmp_path, "cmapss", payload)
    values, problem = ev.load(tmp_path, "cmapss", ev.engine_view)
    assert values is None and problem.kind == "malformed"
    assert detail in problem.message and "`make cmapss`" in problem.message


def test_non_finite_or_boolean_numbers_are_not_displayed(tmp_path):
    data = json.loads((ROOT / "evidence/cmapss_fd001.json").read_text())
    data["official_test"]["rmse"] = math.nan
    write(tmp_path, "cmapss", data)
    assert ev.load(tmp_path, "cmapss", ev.engine_view)[1].kind == "malformed"
    data["official_test"]["rmse"] = 17.0
    data["official_test"]["n"] = True
    write(tmp_path, "cmapss", data)
    assert "official_test.n" in ev.load(tmp_path, "cmapss", ev.engine_view)[1].message


def test_committed_engine_evidence_values_pass_through_unchanged(tmp_path):
    shutil.copy(ROOT / "evidence/cmapss_fd001.json", tmp_path / "cmapss.json")
    data = json.loads((tmp_path / "cmapss.json").read_text())
    values = ev.engine_view(data)
    assert values["n"] == data["official_test"]["n"]
    assert values["rmse"] == data["official_test"]["rmse"]
    assert values["interval"] == data["official_test_ci_95"]["rmse_95"]
    assert [row["candidate"] for row in values["validation"]] == list(data["validation"])


def test_fractions_outside_unit_interval_are_malformed(tmp_path):
    write(tmp_path, "utilities", {"anomaly": {"injected_anomaly_rows": 30,
                                              "precision": 1.5, "recall": 0.9}})
    problem = ev.load(tmp_path, "utilities", ev.utilities_view)[1]
    assert problem.kind == "malformed" and "precision" in problem.message


def test_directory_in_place_of_file_is_malformed(tmp_path):
    ev.artifact_path(tmp_path, "photo").mkdir(parents=True)
    assert ev.load(tmp_path, "photo", ev.photo_view)[1].detail == "not a regular file"


def test_database_problems_are_reported(tmp_path):
    assert ev.load_database(tmp_path)[1].kind == "missing"
    write(tmp_path, "database", "this is not a DuckDB file")
    problem = ev.load_database(tmp_path)[1]
    assert problem.kind == "malformed" and "`make utilities`" in problem.message
    path = ev.artifact_path(tmp_path, "database")
    path.unlink()
    duckdb.connect(str(path)).close()  # valid file, missing table
    assert "DuckDB could not read it" in ev.load_database(tmp_path)[1].message


def test_artifact_root_override(monkeypatch, tmp_path):
    monkeypatch.delenv(ev.ROOT_ENV, raising=False)
    assert ev.artifact_root() == ev.PROJECT_ROOT
    monkeypatch.setenv(ev.ROOT_ENV, str(tmp_path))
    assert ev.artifact_root() == tmp_path.resolve()


def test_markdown_table_has_no_index_and_escapes_cells():
    text = ev.markdown_table([{"candidate": "constant_median", "RMSE": "56.66"},
                              {"candidate": "a|b", "RMSE": "1.00"}])
    lines = text.splitlines()
    assert lines[0] == "| candidate | RMSE |" and lines[1] == "|---|---|"
    assert lines[2] == "| constant\\_median | 56.66 |" and lines[3] == "| a\\|b | 1.00 |"
    with pytest.raises(ValueError):
        ev.markdown_table([])
