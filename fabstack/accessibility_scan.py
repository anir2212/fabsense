"""Automated axe-core scan of the five dashboard tabs in headless Chrome.

Reuses the browser walkthrough's server and Chrome setup. For each tab it
injects axe-core (the copy bundled in ``axe-playwright-python``, MPL-2.0),
runs the WCAG 2.0/2.1 A and AA rule tags on the whole document, and records
violations and "needs review" (incomplete) items with impact, node counts and
sample selectors. Writes ``evidence/accessibility_scan.json``.

An automated scan finds only a subset of accessibility problems. It is not a
WCAG conformance claim and does not replace a manual screen-reader review.
Node targets inside Streamlit's own page chrome (header, toolbar, main menu)
or the Vega chart action menu are labelled ``framework`` by a selector/HTML
heuristic; everything else, including Streamlit widgets the app chose, is ``app``.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from fabstack import browser_walkthrough as bw
from fabstack import dashboard_evidence as ev
from fabstack.boundary_check import expected_metrics

ROOT = ev.PROJECT_ROOT
RESULT = ROOT / "evidence/accessibility_scan.json"
TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]
LOCAL_PATH = re.compile(r"(/private)?/var/folders/[^\s<\"']*|/Users/[^/\s<\"']+")
FRAMEWORK_MARKERS = ("stHeader", "stToolbar", "stMainMenu", "MainMenu", "stAppDeployButton",
                     "stDeployButton", "stBaseButton-header", "stDecoration", "stStatusWidget",
                     "Click to view actions", "vega-actions")


def axe_script() -> Path | None:
    try:
        import axe_playwright_python
    except ImportError:
        return None
    path = Path(axe_playwright_python.__file__).with_name("axe.min.js")
    return path if path.is_file() else None


def axe_version(path: Path) -> str:
    first = path.read_text(encoding="utf-8", errors="replace")[:200]
    return first.split("axe v", 1)[1].split()[0] if "axe v" in first else "unknown"


def redact(html: str) -> str:
    """Hide temporary-folder and home paths so evidence can be shared."""
    return LOCAL_PATH.sub("<local-path>", html)


def origin(target: list, html: str = "") -> str:
    text = " ".join(str(part) for part in target) + " " + html
    return "framework" if any(marker in text for marker in FRAMEWORK_MARKERS) else "app"


def summarize(results: dict) -> dict:
    """Reduce one axe result to rule-level counts and a few sample selectors."""
    def rules(items: list) -> list[dict]:
        out = []
        for item in items:
            nodes = item.get("nodes", [])
            origins = [origin(node.get("target", []), node.get("html", "")) for node in nodes]
            out.append({"id": item["id"], "impact": item.get("impact"),
                        "help": item.get("help"), "help_url": item.get("helpUrl"),
                        "nodes": len(nodes), "app_nodes": origins.count("app"),
                        "framework_nodes": origins.count("framework"),
                        "sample_targets": [node.get("target") for node in nodes[:3]],
                        "sample_html": [redact(node.get("html", ""))[:200] for node in nodes[:3]]})
        return sorted(out, key=lambda rule: rule["id"])

    violations = rules(results.get("violations", []))
    return {"violations": violations, "needs_review": rules(results.get("incomplete", [])),
            "passed_rules": len(results.get("passes", [])),
            "inapplicable_rules": len(results.get("inapplicable", [])),
            "violation_nodes": sum(rule["nodes"] for rule in violations),
            "app_violation_nodes": sum(rule["app_nodes"] for rule in violations)}


def run_axe(page, script: Path) -> dict:
    if not page.evaluate("() => typeof window.axe !== 'undefined'"):
        page.add_script_tag(path=str(script))
    return page.evaluate(
        "(tags) => axe.run(document, {runOnly: {type: 'tag', values: tags}})", TAGS)


def scan(browser, artifact_root: Path | None, workdir: Path, script: Path,
         label: str, metric_count: int, alert_count: int) -> dict:
    tabs = {}
    with bw.streamlit_server(artifact_root, workdir / f"{label}.log") as (url, server):
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url)
        bw.wait_for_render(page, metric_count, alert_count)
        for name in bw.TABS:
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(1200)  # let charts and tables finish rendering
            tabs[name] = summarize(run_axe(page, script))
        page.close()
    return {"artifact_root": label, "tabs": tabs, "server_stopped": server["stopped_cleanly"]}


def aggregate(scenarios: dict) -> dict:
    rules: dict[str, dict] = {}
    for scenario, result in scenarios.items():
        for tab, summary in result["tabs"].items():
            for rule in summary["violations"]:
                entry = rules.setdefault(rule["id"], {
                    "impact": rule["impact"], "help": rule["help"], "where": [],
                    "max_nodes": 0, "app_nodes_seen": False})
                entry["where"].append(f"{scenario}/{tab}")
                entry["max_nodes"] = max(entry["max_nodes"], rule["nodes"])
                entry["app_nodes_seen"] |= rule["app_nodes"] > 0
    return dict(sorted(rules.items()))


def main() -> int:
    reason = bw.skip_reason()
    script = axe_script()
    if reason or script is None:
        print(f"SKIP accessibility scan: {reason or 'install requirements-browser.txt (axe-core)'}")
        return 0
    from playwright.sync_api import sync_playwright

    started = datetime.now(UTC)
    generated = all(ev.artifact_path(ROOT, key).is_file() for key in ev.ARTIFACTS)
    with tempfile.TemporaryDirectory(prefix="fabsense-a11y-") as temporary, \
            sync_playwright() as playwright:
        workdir = Path(temporary)
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        try:
            empty = workdir / "empty_artifacts"
            empty.mkdir()
            scenarios = {"empty_evidence": scan(browser, empty, workdir, script,
                                                "empty_evidence", 0, 1)}
            if generated:
                scenarios["full_evidence"] = scan(browser, None, workdir, script,
                                                  "full_evidence", len(expected_metrics(ROOT)), 0)
            chrome = browser.version
        finally:
            browser.close()
    rules = aggregate(scenarios)
    report = {
        "started_at_utc": started.isoformat(timespec="seconds"),
        "scope": "Automated axe-core scan (WCAG 2.0/2.1 A and AA tags) of each tab in headless "
                 "Chrome on 127.0.0.1. Not a conformance claim; no screen reader was used.",
        "versions": {"axe_core": axe_version(script), "python": platform.python_version(),
                     "platform": platform.platform(), "chrome": chrome,
                     "streamlit": metadata.version("streamlit"),
                     "playwright": metadata.version("playwright")},
        "tags": TAGS,
        "full_evidence_skipped": None if generated else "Generated runs/ artifacts absent.",
        "violated_rules": rules,
        "violated_rule_count": len(rules),
        "rules_with_app_nodes": sorted(rule for rule, entry in rules.items()
                                       if entry["app_nodes_seen"]),
        "scenarios": scenarios,
        "servers_stopped": all(s["server_stopped"] for s in scenarios.values()),
    }
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"violated_rules": {rule: entry["impact"] for rule, entry in rules.items()},
                      "rules_with_app_nodes": report["rules_with_app_nodes"],
                      "servers_stopped": report["servers_stopped"],
                      "evidence": str(RESULT.relative_to(ROOT))}, indent=2))
    return 0 if report["servers_stopped"] else 1


if __name__ == "__main__":
    sys.exit(main())
