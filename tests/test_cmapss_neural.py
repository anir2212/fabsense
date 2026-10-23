"""FD003 loader checks, window causality, seed determinism and development-only selection."""

import numpy as np
import pandas as pd
import pytest

from fabstack import cmapss, cmapss_neural


def engine_frame(lifetimes, seed=0):
    """Ordered engine/cycle rows with all 26 C-MAPSS columns and a wear trend in the sensors."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, lifetime in enumerate(lifetimes, start=1):
        cycles = np.arange(1, lifetime + 1)
        block = pd.DataFrame(0.0, index=range(lifetime), columns=cmapss.COLUMNS)
        block["unit"] = unit
        block["cycle"] = cycles
        for sensor in cmapss.SENSORS:
            block[sensor] = cycles / lifetime + rng.normal(0, 0.05, lifetime)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def lifetimes_summing_to(total, engines=100):
    base, extra = divmod(total, engines)
    return [base + (1 if index < extra else 0) for index in range(engines)]


def fd003_shaped_tables():
    train = engine_frame(lifetimes_summing_to(24720))
    test = engine_frame(lifetimes_summing_to(16596))
    rul = pd.DataFrame(np.arange(100) + 5)
    return train, test, rul


def test_fd003_loader_accepts_expected_shape_and_rejects_bad_tables():
    train, test, rul = fd003_shaped_tables()
    _, _, labels = cmapss.validate_tables("FD003", train.copy(), test.copy(), rul.copy())
    assert labels.shape == (100,)

    with pytest.raises(ValueError, match="Unexpected FD003 table shape"):
        cmapss.validate_tables("FD003", train.iloc[:-1].copy(), test.copy(), rul.copy())
    wrong_ids = train.copy()
    wrong_ids.loc[wrong_ids["unit"] == 100, "unit"] = 101
    with pytest.raises(ValueError, match="unexpected FD003 unit IDs"):
        cmapss.validate_tables("FD003", wrong_ids, test.copy(), rul.copy())
    gap = test.copy()
    gap.loc[5, "cycle"] = 99
    with pytest.raises(ValueError, match="consecutive"):
        cmapss.validate_tables("FD003", train.copy(), gap, rul.copy())
    negative = rul.copy()
    negative.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="Negative"):
        cmapss.validate_tables("FD003", train.copy(), test.copy(), negative)
    with pytest.raises(ValueError, match="No checked shapes"):
        cmapss.load_subset("FD002")


@pytest.mark.skipif(not cmapss.ARCHIVE.exists(), reason="raw NASA archive not downloaded")
def test_real_fd003_files_pass_loader_checks():
    train, test, labels = cmapss.load_subset("FD003")
    assert train.shape == (24720, 26) and test.shape == (16596, 26) and len(labels) == 100


def test_window_features_use_only_past_cycles_of_the_same_engine():
    frame = engine_frame([6, 4])
    full = cmapss_neural.window_features(frame, 3)
    prefix = cmapss_neural.window_features(frame.iloc[:3].copy(), 3)
    pd.testing.assert_frame_equal(full.iloc[:3], prefix)

    changed = frame.copy()
    changed.loc[5, "sensor_02"] = 1e6  # last cycle of engine 1
    changed.loc[6, "sensor_02"] = -1e6  # first cycle of engine 2
    altered = cmapss_neural.window_features(changed, 3)
    pd.testing.assert_frame_equal(altered.iloc[:5], full.iloc[:5])

    first_value = frame.loc[0, "sensor_02"]
    assert full.loc[0, "sensor_02_lag_02"] == first_value  # edge padding repeats cycle 1
    assert full.loc[3, "sensor_02_lag_02"] == frame.loc[1, "sensor_02"]
    assert full.loc[6, "sensor_02_lag_01"] == frame.loc[6, "sensor_02"]  # engine 2 does not see engine 1
    assert len(full.columns) == 3 * len(cmapss_neural.WINDOW_SENSORS) + 1
    with pytest.raises(ValueError, match="ordered"):
        cmapss_neural.window_features(frame.iloc[::-1], 3)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mlp_is_deterministic_for_a_fixed_seed():
    frame = engine_frame([40] * 10)
    features = cmapss_neural.window_features(frame, 4)
    target = cmapss.targets(frame)
    first = cmapss_neural.mlp((8,), 3).fit(features, target).predict(features)
    second = cmapss_neural.mlp((8,), 3).fit(features, target).predict(features)
    other = cmapss_neural.mlp((8,), 4).fit(features, target).predict(features)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_selection_rules_and_verdicts():
    order = ["a", "b", "c"]
    assert cmapss_neural.select_config({"a": 20.0, "b": 19.0, "c": 19.0}, order) == "b"
    assert cmapss_neural.choose_model(20.77, 20.77) == "xgboost"
    assert cmapss_neural.choose_model(20.77, 20.5) == "mlp"
    assert cmapss_neural.verdict([-3.0, -0.1]) == "neural better"
    assert cmapss_neural.verdict([0.2, 4.0]) == "neural worse"
    assert cmapss_neural.verdict([-1.0, 1.0]) == "no demonstrated difference"
    actual = np.array([10.0, 20.0, 30.0])
    delta = cmapss_neural.paired_delta_bootstrap(actual, actual + 5, np.vstack([actual, actual + 1]))
    assert delta["point"] == pytest.approx(0.5 - 5.0)
    assert delta["interval_95"][1] < 0


def test_selection_uses_development_data_only(monkeypatch):
    """Changing every FD003 label and reading must leave the development stage unchanged."""
    monkeypatch.setattr(cmapss_neural, "SAVED_FD001", cmapss_neural.Path("missing.json"))
    development = engine_frame([30 + unit % 7 for unit in range(100)], seed=1)
    calls = []

    def load_development():
        calls.append("development")
        return development, None, None

    def make_test(seed, shift):
        def load_test():
            calls.append("test")
            test = engine_frame([12 + unit % 5 for unit in range(100)], seed=seed)
            return None, test, np.arange(100) + shift

        return load_test

    grid = [(3, (4,)), (5, (4,))]
    one = cmapss_neural.run(load_development, make_test(2, 0), grid=grid, seeds=(0, 1))
    two = cmapss_neural.run(load_development, make_test(3, 40), grid=grid, seeds=(0, 1))
    assert calls == ["development", "test"] * 2

    def strip(development_block):
        """Drop wall-clock fit times, which legitimately vary between runs."""
        clean = {"baseline": {k: v for k, v in development_block["baseline"].items() if k != "fit"}}
        clean["mlp"] = {
            name: [{k: v for k, v in row.items() if k != "fit_seconds"} for row in block["per_seed"]]
            for name, block in development_block["mlp"].items()
        }
        return clean

    assert strip(one["development"]) == strip(two["development"])
    assert one["selection"] == two["selection"]
    assert one["fd003_test"]["baseline"]["rmse"] != two["fd003_test"]["baseline"]["rmse"]
    assert one["data_use"]["fd001_official_test"] == "not scored in this study"
