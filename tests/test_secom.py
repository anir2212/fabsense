"""Guard the SECOM baseline's fold-local preprocessing and threshold selection."""

import numpy as np
import pytest

from fabstack.secom import MissingnessFilter, preprocessor, threshold_for_precision


def test_preprocessing_learns_missingness_and_median_from_fit_rows_only():
    train = np.array([
        [np.nan, 1.0, 0.0],
        [np.nan, 2.0, 1.0],
        [np.nan, np.nan, 0.0],
        [1.0, 4.0, 1.0],
    ])
    future = np.array([[999.0, np.nan, 1.0]])
    pipeline = preprocessor().fit(train)
    assert pipeline.named_steps["missing"].keep_.tolist() == [False, True, True]
    transformed = pipeline.transform(future)
    assert transformed.shape[0] == 1
    assert transformed[0, 0] == 2.0  # train median, never the future value


def test_missingness_filter_rejects_changed_sensor_count():
    fitted = MissingnessFilter().fit(np.array([[1.0, 2.0], [2.0, np.nan]]))
    with pytest.raises(ValueError, match="Sensor count changed"):
        fitted.transform(np.array([[1.0, 2.0, 3.0]]))


def test_precision_threshold_is_selected_from_development_scores():
    label = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    assert threshold_for_precision(label, score, minimum=0.8) == pytest.approx(0.9)
    assert threshold_for_precision(label, score, minimum=1.01) == 1.0
