"""Render the Streamlit dashboard in process with complete, missing and malformed evidence."""

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fabstack import dashboard_evidence as ev

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "dashboards/streamlit_app.py")
TABS = ["Photo bay", "Sustainability", "Workstation MSA", "Engine remaining life", "Ask the data"]
# Evidence each tab needs, in tab order.
TAB_EVIDENCE = [
    ["runs/photo_grid.json", "figures/photo_tradeoff.png"],
    ["runs/utilities/results.json", "runs/utilities/fabsense.duckdb"],
    ["runs/msa/results.json", "figures/msa_agreement.png"],
    ["evidence/cmapss_fd001.json"],
    ["runs/utilities/fabsense.duckdb"],
]


def run_app(monkeypatch, root: Path | None) -> AppTest:
    if root is None:
        monkeypatch.delenv(ev.ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(ev.ROOT_ENV, str(root))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, [error.message for error in app.exception]
    assert [tab.label for tab in app.tabs] == TABS
    return app


def messages(element) -> str:
    return " ".join(item.value for item in [*element.warning, *element.error])


def engine_expected() -> dict:
    data = json.loads((ROOT / "evidence/cmapss_fd001.json").read_text())
    test = data["official_test"]
    return {"Official test engines": str(test["n"]), "RMSE (cycles)": f"{test['rmse']:.2f}",
            "MAE (cycles)": f"{test['mae']:.2f}"}


def test_empty_artifact_root_degrades_every_tab(monkeypatch, tmp_path):
    app = run_app(monkeypatch, tmp_path)
    assert not app.metric  # no numbers without evidence
    assert "8 evidence file(s) are missing or malformed" in app.warning[0].value
    for tab, paths in zip(app.tabs, TAB_EVIDENCE):
        text = messages(tab)
        for path in paths:
            assert f"Missing evidence file `{path}`" in text
        assert "Regenerate it with `make " in text
    assert "`make cmapss`" in messages(app.tabs[3])


def test_engine_tab_reads_committed_evidence_alone(monkeypatch, tmp_path):
    (tmp_path / "evidence").mkdir()
    shutil.copy(ROOT / "evidence/cmapss_fd001.json", tmp_path / "evidence")
    app = run_app(monkeypatch, tmp_path)
    engine = app.tabs[3]
    assert {metric.label: metric.value for metric in engine.metric} == engine_expected()
    assert not messages(engine)
    assert "Missing evidence file `runs/photo_grid.json`" in messages(app.tabs[0])


def test_malformed_evidence_shows_errors_not_values(monkeypatch, tmp_path):
    for key, payload in (("cmapss", "{truncated"), ("photo", json.dumps({"scenarios": []})),
                         ("utilities", json.dumps({"anomaly": {"precision": "high"}})),
                         ("database", "not a database")):
        path = ev.artifact_path(tmp_path, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    app = run_app(monkeypatch, tmp_path)
    assert not app.metric
    assert "Malformed evidence file `evidence/cmapss_fd001.json` (not valid JSON" in \
        app.tabs[3].error[0].value
    assert "Malformed evidence file `runs/photo_grid.json`" in messages(app.tabs[0])
    assert "Malformed evidence file `runs/utilities/results.json`" in messages(app.tabs[1])
    assert "Malformed evidence file `runs/utilities/fabsense.duckdb`" in messages(app.tabs[4])


def test_full_project_dashboard_matches_saved_json(monkeypatch):
    if not all((ROOT / path).exists() for paths in TAB_EVIDENCE for path in paths):
        pytest.skip("Generated runs/ artifacts are absent; run `make all` first.")
    app = run_app(monkeypatch, None)
    assert not app.warning and not app.error
    utilities = json.loads((ROOT / "runs/utilities/results.json").read_text())["anomaly"]
    msa = json.loads((ROOT / "runs/msa/results.json").read_text())["workstations"]
    expected = {
        "Injected anomaly rows": str(utilities["injected_anomaly_rows"]),
        "Precision": f"{utilities['precision']:.1%}",
        "Recall": f"{utilities['recall']:.1%}",
        "Old %GRR": f"{msa['old']['anova']['percent_grr']:.2f}%",
        "New %GRR": f"{msa['new']['anova']['percent_grr']:.2f}%",
        "New − old bias": f"{msa['agreement']['new_minus_old_mean']:.3f}",
        **engine_expected(),
    }
    assert {metric.label: metric.value for metric in app.metric} == expected
    app.tabs[4].button[0].click().run()
    assert app.tabs[4].json[0].value  # citations rendered
