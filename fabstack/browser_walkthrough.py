"""Real-browser walkthrough of the five dashboard tabs using the installed Google Chrome.

Starts a headless Streamlit server on 127.0.0.1 with usage statistics disabled,
drives it with Playwright (``channel="chrome"``; no browser download), and
writes compact evidence to ``evidence/browser_walkthrough.json``. Two runs are
checked: the project's real artifacts and an empty artifact root. When Chrome
or Playwright is unavailable the command prints SKIP and exits 0 without
writing evidence. This is a local check, not a production or accessibility audit.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from fabstack import dashboard_evidence as ev
from fabstack.boundary_check import expected_metrics

ROOT = ev.PROJECT_ROOT
RESULT = ROOT / "evidence/browser_walkthrough.json"
SHOTS = ROOT / "evidence/browser"
TABS = ["Photo bay", "Sustainability", "Workstation MSA", "Engine remaining life", "Ask the data"]
HEADERS = ["Scanner train-size scenarios", "Weather-adjusted meter exceptions",
           "Old versus new workstation", "Simulated engine remaining life", "Ask the local data"]
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
QUESTION = "Which building had the highest electricity excess?"


def chrome_path() -> Path | None:
    candidates = [os.environ.get("CHROME_PATH"), str(MAC_CHROME),
                  shutil.which("google-chrome"), shutil.which("google-chrome-stable")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def skip_reason() -> str | None:
    if chrome_path() is None:
        return "Google Chrome was not found (set CHROME_PATH to its executable)."
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return "Playwright is not installed; run `.venv/bin/python -m pip install -r requirements-browser.txt`."
    return None


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@contextmanager
def streamlit_server(artifact_root: Path | None, log_path: Path):
    """Run Streamlit headless on a free localhost port and always stop it afterwards."""
    port = free_port()
    env = {**os.environ, "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    env.pop(ev.ROOT_ENV, None)
    if artifact_root is not None:
        env[ev.ROOT_ENV] = str(artifact_root)
    command = [sys.executable, "-m", "streamlit", "run", "dashboards/streamlit_app.py",
               "--server.address", "127.0.0.1", "--server.port", str(port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    server = {"port": port, "stopped_cleanly": None}
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited early; see {log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                                timeout=1) as response:
                        if response.read() == b"ok":
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Streamlit did not become healthy within 60 s")
                time.sleep(0.3)
            yield f"http://127.0.0.1:{port}/", server
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            server["stopped_cleanly"] = process.returncode is not None and not port_open(port)


def tab_state(page) -> dict:
    return page.evaluate("""() => {
        const tabs = [...document.querySelectorAll('[role=tab]')];
        const active = document.activeElement;
        return {
            focused: active && active.getAttribute('role') === 'tab' ? active.innerText.trim() : null,
            selected: tabs.filter(t => t.getAttribute('aria-selected') === 'true').map(t => t.innerText.trim()),
            tabindex: tabs.map(t => t.getAttribute('tabindex')),
        };
    }""")


def visible_panel(page):
    return page.locator('[role=tabpanel]:not([hidden])')


def panel_snapshot(page) -> dict:
    panel = visible_panel(page)
    metrics = {}
    for metric in panel.locator('[data-testid="stMetric"]').all():
        label = metric.locator('[data-testid="stMetricLabel"]').inner_text().strip()
        metrics[label] = metric.locator('[data-testid="stMetricValue"]').inner_text().strip()
    return {
        "header": panel.locator("h2").first.inner_text().strip() if panel.locator("h2").count() else None,
        "metrics": metrics,
        "alerts": [{"kind": alert.get_attribute("data-testid").removeprefix("stAlertContent").lower(),
                    "text": alert.inner_text().strip()}
                   for alert in panel.locator('[data-testid^="stAlertContent"]').all()],
    }


TABLE_SELECTOR = "[role=tabpanel]:not([hidden]) [data-testid=stMarkdownContainer] table"


def read_tables(page, expected: int) -> list[list[list[str]]]:
    """Header and body cell text of each static table in the visible tab.

    Waits until ``expected`` tables are attached, so a slow render is not misread.
    """
    page.wait_for_function("([selector, count]) => document.querySelectorAll(selector).length >= count",
                           arg=[TABLE_SELECTOR, expected], timeout=30000)
    return page.evaluate("""(selector) => [...document.querySelectorAll(selector)].map(table =>
        [...table.querySelectorAll('tr')].map(row =>
            [...row.querySelectorAll('th, td')].map(cell => cell.innerText.trim())))""", TABLE_SELECTOR)


def wait_for_render(page, metric_count: int, alert_count: int) -> None:
    page.get_by_role("tab").first.wait_for(timeout=30000)
    page.wait_for_function(
        """([metrics, alerts]) =>
            document.querySelectorAll('[data-testid="stMetric"]').length === metrics &&
            document.querySelectorAll('[data-testid="stAlert"]').length >= alerts""",
        arg=[metric_count, alert_count], timeout=30000)


def keyboard_walk(page) -> dict:
    """Arrow keys between tabs, with focus following selection (WAI-ARIA tabs pattern)."""
    tabs = page.get_by_role("tab")
    tabs.first.click()
    tabs.first.focus()
    steps = [{"key": "(focus first tab)", **tab_state(page),
              "panel_header": panel_snapshot(page)["header"]}]
    # After ArrowLeft focus is on the last tab: test Home from there, then End from a middle tab.
    for key in ["ArrowRight"] * 5 + ["ArrowLeft", "Home", "ArrowRight", "ArrowLeft", "ArrowLeft", "End"]:
        page.keyboard.press(key)
        page.wait_for_timeout(250)
        steps.append({"key": key, **tab_state(page), "panel_header": panel_snapshot(page)["header"]})
    right = steps[1:6]
    expected_right = TABS[1:] + TABS[:1]
    checks = {
        "arrow_right_moves_selection_and_focus": all(
            step["selected"] == [name] and step["focused"] == name
            for step, name in zip(right, expected_right)),
        "arrow_right_wraps_last_to_first": right[-1]["selected"] == [TABS[0]],
        "arrow_left_wraps_first_to_last": steps[6]["selected"] == [TABS[-1]],
        "panel_follows_selected_tab": all(
            step["panel_header"] == HEADERS[TABS.index(step["selected"][0])] for step in steps),
        "roving_tabindex_single_zero": all(step["tabindex"].count("0") == 1 for step in steps),
    }
    middle = steps[10]["selected"]
    observed = {
        "home_from_last_tab_selected": steps[7]["selected"],
        "home_moves_to_first_tab": steps[7]["selected"] == [TABS[0]],
        "end_pressed_from": middle,
        "end_selected": steps[11]["selected"],
        "end_moves_to_last_tab": middle != [TABS[-1]] and steps[11]["selected"] == [TABS[-1]],
    }
    return {"steps": steps, "checks": checks, "observed_not_required": observed}


def ask_with_keyboard(page) -> dict:
    """Select the question tab and submit the default question without the mouse."""
    from fabstack.agent import AnalyticsAgent

    page.get_by_role("tab", name="Ask the data").focus()
    page.keyboard.press("Enter")
    question = visible_panel(page).get_by_label("Question")
    question.focus()
    page.keyboard.press("Tab")
    focused = page.evaluate("() => document.activeElement.innerText.trim()")
    page.keyboard.press("Enter")
    expected = AnalyticsAgent(ev.artifact_path(ROOT, "database")).answer(QUESTION)
    page.get_by_text(expected["answer"]).wait_for(timeout=30000)
    return {"question": QUESTION, "tab_moved_focus_to": focused,
            "answer_shown": expected["answer"],
            "cited_query_id": expected["citations"][0]["query_id"],
            "cited_row_count": len(expected["citations"][0]["row_ids"]),
            "ok": focused == "Run question"}


def full_scenario(browser, workdir: Path) -> dict:
    expected = expected_metrics(ROOT)
    engine, _ = ev.load(ROOT, "cmapss", ev.engine_view)
    interval = f"[{engine['interval'][0]:.2f}, {engine['interval'][1]:.2f}]"
    tables = ev.engine_tables(engine)
    tables_expected = [[list(rows[0])] + [list(row.values()) for row in rows]
                       for rows in (tables["validation"], tables["slices"])]
    with streamlit_server(None, workdir / "full.log") as (url, server):
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url)
        wait_for_render(page, len(expected), 0)
        tabs = page.get_by_role("tab")
        names = [tabs.nth(index).inner_text().strip() for index in range(tabs.count())]
        panels, shown = [], {}
        for name in TABS:
            page.get_by_role("tab", name=name).click()
            snapshot = panel_snapshot(page)
            panels.append({"tab": name, **snapshot})
            shown.update(snapshot["metrics"])
            if name == "Engine remaining life":
                engine_caption_has_interval = visible_panel(page).get_by_text(interval).count() > 0
                tables_shown = read_tables(page, len(tables_expected))
                page.wait_for_timeout(1500)  # let the layout settle after the tab becomes visible
                SHOTS.mkdir(exist_ok=True)
                page.screenshot(path=str(SHOTS / "engine_tab.jpg"), type="jpeg", quality=60)
        exceptions = page.locator('[data-testid="stException"]').count()
        keyboard = keyboard_walk(page)
        question = ask_with_keyboard(page)
        page.close()
    checks = {
        "five_tabs_in_order": names == TABS,
        "each_tab_header_visible": [panel["header"] for panel in panels] == HEADERS,
        "no_warnings_or_errors_with_complete_evidence": all(
            alert["kind"] == "info" for panel in panels for alert in panel["alerts"]),
        "metrics_equal_saved_json": shown == expected,
        "engine_bootstrap_interval_matches_json": engine_caption_has_interval,
        "engine_tables_equal_saved_json": tables_shown == tables_expected,
        "no_script_exception": exceptions == 0,
        "keyboard_question_submitted": question["ok"],
        **keyboard["checks"],
        "server_stopped": server["stopped_cleanly"],
    }
    return {"artifact_root": "project", "tabs_found": names, "panels": panels,
            "metrics_expected_from_json": expected, "metrics_read_from_page": shown,
            "engine_interval_text": interval, "engine_tables_read_from_page": tables_shown,
            "keyboard": keyboard, "keyboard_question": question,
            "screenshot": "evidence/browser/engine_tab.jpg", "checks": checks}


def empty_scenario(browser, workdir: Path) -> dict:
    empty = workdir / "empty_artifacts"
    empty.mkdir()
    with streamlit_server(empty, workdir / "empty.log") as (url, server):
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url)
        wait_for_render(page, 0, 1)
        tabs = page.get_by_role("tab")
        names = [tabs.nth(index).inner_text().strip() for index in range(tabs.count())]
        panels = []
        for name in TABS:
            page.get_by_role("tab", name=name).click()
            panels.append({"tab": name, **panel_snapshot(page)})
            if name == "Engine remaining life":
                SHOTS.mkdir(exist_ok=True)
                page.screenshot(path=str(SHOTS / "empty_engine_tab.jpg"), type="jpeg", quality=60)
        metric_elements = page.locator('[data-testid="stMetric"]').count()
        exceptions = page.locator('[data-testid="stException"]').count()
        keyboard = keyboard_walk(page)
        page.close()
    checks = {
        "five_tabs_in_order": names == TABS,
        "each_tab_header_visible": [panel["header"] for panel in panels] == HEADERS,
        "every_tab_names_missing_file_and_command": all(
            any(alert["kind"] == "warning" and "Missing evidence file" in alert["text"]
                and "Regenerate it with make " in alert["text"] for alert in panel["alerts"])
            for panel in panels),
        "no_metric_values_without_evidence": metric_elements == 0,
        "no_script_exception": exceptions == 0,
        **keyboard["checks"],
        "server_stopped": server["stopped_cleanly"],
    }
    return {"artifact_root": "empty temporary directory", "tabs_found": names, "panels": panels,
            "metric_elements": metric_elements, "keyboard": keyboard,
            "screenshot": "evidence/browser/empty_engine_tab.jpg", "checks": checks}


def main() -> int:
    reason = skip_reason()
    if reason:
        print(f"SKIP browser walkthrough: {reason}")
        return 0
    from playwright.sync_api import sync_playwright

    started = datetime.now(UTC)
    generated = all(ev.artifact_path(ROOT, key).is_file() for key in ev.ARTIFACTS)
    with tempfile.TemporaryDirectory(prefix="fabsense-browser-") as temporary, sync_playwright() as p:
        workdir = Path(temporary)
        launch = ({"executable_path": os.environ["CHROME_PATH"]} if os.environ.get("CHROME_PATH")
                  else {"channel": "chrome"})
        browser = p.chromium.launch(headless=True, **launch)
        try:
            scenarios = {"empty_evidence": empty_scenario(browser, workdir)}
            if generated:
                scenarios["full_evidence"] = full_scenario(browser, workdir)
            chrome_version = browser.version
        finally:
            browser.close()
    report = {
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "Local headless Chrome check on 127.0.0.1; not a production, load or full accessibility audit.",
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     "streamlit": metadata.version("streamlit"),
                     "playwright": metadata.version("playwright"), "chrome": chrome_version},
        "full_evidence_skipped": None if generated else "Generated runs/ artifacts absent; run `make all`.",
        "scenarios": scenarios,
    }
    report["checks_passed"] = sum(value is True for s in scenarios.values() for value in s["checks"].values())
    report["checks_total"] = sum(len(s["checks"]) for s in scenarios.values())
    report["ok"] = report["checks_passed"] == report["checks_total"]
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": report["ok"], "checks": f"{report['checks_passed']}/{report['checks_total']}",
                      "evidence": str(RESULT.relative_to(ROOT))}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
