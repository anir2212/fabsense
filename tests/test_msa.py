import pytest

from fabstack.msa import crossed_anova, generate, spc_drift, workstation_report


def test_known_bias_and_gauge_variation_recovered():
    report = workstation_report(generate(seed=7))
    old, new = report["old"], report["new"]
    assert old["observed_bias"] == pytest.approx(0.15, abs=0.03)
    assert new["observed_bias"] == pytest.approx(0.03, abs=0.03)
    assert new["anova"]["gauge_rr_sd"] < old["anova"]["gauge_rr_sd"]
    assert new["anova"]["percent_grr"] < old["anova"]["percent_grr"]
    assert report["agreement"]["new_minus_old_mean"] == pytest.approx(-0.12, abs=0.03)


def test_crossed_design_rejects_missing_measurement():
    rows = [row for row in generate(seed=7) if row.workstation == "old"]
    with pytest.raises(ValueError, match="balanced"):
        crossed_anova(rows[:-1])


def test_planted_drift_is_detected_after_change():
    result = spc_drift(seed=7)
    assert result["ewma_first_signal"] is not None
    assert result["cusum_first_signal"] is not None
    assert result["ewma_first_signal"] >= 50
