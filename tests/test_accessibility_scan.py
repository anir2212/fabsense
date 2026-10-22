"""Pure helpers of the axe-core scan (no browser needed)."""

from fabstack.accessibility_scan import aggregate, origin, redact, summarize


def test_origin_separates_streamlit_chrome_from_app_content():
    assert origin(["#MainMenu"]) == "framework"
    assert origin([".x"], '<button data-testid="stBaseButton-headerNoPadding">') == "framework"
    assert origin(['details[title="Click to view actions"] > summary']) == "framework"
    assert origin([".stCaptionContainer > p"], "<p>Seeded data</p>") == "app"


def test_summarize_counts_nodes_by_origin_and_keeps_samples():
    result = {
        "violations": [{"id": "color-contrast", "impact": "serious", "help": "h", "helpUrl": "u",
                        "nodes": [{"target": [".a"], "html": "<p>x</p>"},
                                  {"target": ["#MainMenu"], "html": "<span>"}]}],
        "incomplete": [{"id": "aria-hidden-focus", "nodes": [{"target": [".b"]}]}],
        "passes": [{}, {}], "inapplicable": [{}],
    }
    summary = summarize(result)
    rule = summary["violations"][0]
    assert (rule["nodes"], rule["app_nodes"], rule["framework_nodes"]) == (2, 1, 1)
    assert rule["sample_html"] == ["<p>x</p>", "<span>"]
    assert summary["needs_review"][0]["id"] == "aria-hidden-focus"
    assert (summary["passed_rules"], summary["inapplicable_rules"]) == (2, 1)
    assert (summary["violation_nodes"], summary["app_violation_nodes"]) == (2, 1)


def test_aggregate_lists_where_each_rule_fails():
    tab = {"violations": [{"id": "button-name", "impact": "critical", "help": "h",
                           "nodes": 1, "app_nodes": 0}]}
    rules = aggregate({"full": {"tabs": {"A": tab, "B": tab}}})
    assert rules["button-name"]["where"] == ["full/A", "full/B"]
    assert rules["button-name"]["app_nodes_seen"] is False


def test_redact_hides_local_paths():
    html = "<p>root: /private/var/folders/1b/abc/T/x and /Users/someone/project</p>"
    assert redact(html) == "<p>root: <local-path> and <local-path>/project</p>"
