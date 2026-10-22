"""The public-data boundary check must detect hash and catalog drift without downloads."""

import json
import shutil
from pathlib import Path

from fabstack import boundary_check as bc

ROOT = Path(__file__).resolve().parents[1]


def copy_contract_files(destination: Path) -> None:
    for name in ("data/SOURCES.md", "warehouse/QUERY_CATALOG.md", "fabstack/secom.py",
                 "fabstack/cmapss.py", "fabstack/agent.py", "dashboards/streamlit_app.py",
                 "evidence/secom_baseline.json", "evidence/cmapss_fd001.json"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, target)


def test_parsers_read_constants_hashes_and_catalog_rows():
    assert bc.module_hash('URL = "x"\nSHA256 = "' + "a" * 64 + '"\n') == "a" * 64
    assert bc.documented_hashes("hash `" + "b" * 64 + "` and short abc") == {"b" * 64}
    ids = bc.catalog_ids((ROOT / "warehouse/QUERY_CATALOG.md").read_text())
    assert {"top_building_excess_v1", "meter_series_by_building_v1"} <= ids


def test_absent_archives_are_not_downloaded_and_documented_hashes_agree(tmp_path):
    copy_contract_files(tmp_path)
    report = bc.check_archives(tmp_path)
    assert all(not entry["present_locally"] and entry["ok"] for entry in report.values())
    assert not (tmp_path / "data/raw").exists()


def test_corrupt_local_archive_and_undocumented_hash_fail(tmp_path):
    copy_contract_files(tmp_path)
    (tmp_path / "data/raw").mkdir()
    (tmp_path / "data/raw/secom.zip").write_bytes(b"corrupt")
    report = bc.check_archives(tmp_path)
    assert report["uci_secom"]["local_matches_module"] is False
    assert report["uci_secom"]["ok"] is False
    evidence = tmp_path / "evidence/cmapss_fd001.json"
    data = json.loads(evidence.read_text())
    data["source_sha256"] = "0" * 64
    evidence.write_text(json.dumps(data))
    assert bc.check_archives(tmp_path)["nasa_cmapss"]["ok"] is False


def test_undocumented_query_id_fails_without_generated_artifacts(tmp_path):
    copy_contract_files(tmp_path)
    assert bc.check_citations(tmp_path)["ok"] is True
    catalog = tmp_path / "warehouse/QUERY_CATALOG.md"
    catalog.write_text(catalog.read_text().replace("`top_building_excess_v1`", "`renamed_v1`"))
    report = bc.check_citations(tmp_path)
    assert report["generated_artifacts_present"] is False and report["ok"] is False
