"""FD001 feature and metric rules that could otherwise leak future cycles."""

import numpy as np
import pandas as pd
import pytest

from fabstack import cmapss


def small_frame():
    rows = []
    for unit, readings in ((1, [1.0, 2.0, 100.0]), (2, [8.0, 9.0, 10.0])):
        for cycle, value in enumerate(readings, start=1):
            row = {name: 0.0 for name in cmapss.COLUMNS}
            row.update(unit=unit, cycle=cycle, sensor_01=value)
            rows.append(row)
    return pd.DataFrame(rows, columns=cmapss.COLUMNS)


def test_rolling_features_do_not_see_future_or_other_engines():
    frame = small_frame()
    full = cmapss.causal_features(frame)
    prefix = cmapss.causal_features(frame.iloc[:2].copy())
    pd.testing.assert_series_equal(full.loc[1], prefix.loc[1])
    assert full.loc[3, "sensor_01_mean_5"] == 8.0
    assert full.loc[2, "sensor_01_mean_5"] == pytest.approx((1 + 2 + 100) / 3)


def test_validation_endpoints_and_rul_targets_are_engine_local():
    frame = small_frame()
    assert cmapss.targets(frame).tolist() == [2, 1, 0, 2, 1, 0]
    with pytest.raises(ValueError, match="Duplicate validation endpoints"):
        cmapss.validation_indices(frame, [2])
    longer = pd.DataFrame({"unit": [2] * 20, "cycle": range(1, 21)})
    indices = cmapss.validation_indices(longer, [2])
    assert len(indices) == 4
    assert longer.loc[indices, "unit"].tolist() == [2] * 4


def test_nasa_score_penalizes_late_prediction_more():
    truth = np.array([50.0])
    early = cmapss.score(truth, np.array([40.0]))
    late = cmapss.score(truth, np.array([60.0]))
    assert early["rmse"] == late["rmse"] == 10.0
    assert late["nasa_score_sum"] > early["nasa_score_sum"]


def test_archive_hash_rejects_corrupt_download(tmp_path):
    archive = tmp_path / "cmapss.zip"
    archive.write_bytes(b"not the NASA archive")
    with pytest.raises(ValueError, match="hash mismatch"):
        cmapss.ensure_archive(archive)
