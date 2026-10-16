import pytest

from fabstack.sim_photo import Scenario, simulate


def test_shared_reticles_and_train_size_show_expected_measured_tradeoff():
    low_load = Scenario(1, 0.05)
    small = simulate(low_load, seed=7)
    medium = simulate(Scenario(4, 0.05), seed=7)
    large = simulate(Scenario(24, 0.05), seed=7)
    assert small == simulate(low_load, seed=7)
    assert small["arrived"] == medium["arrived"] == large["arrived"]
    assert small["completed"] + small["backlog"] == small["arrived"]
    assert small["setup_time_share"] > medium["setup_time_share"] > large["setup_time_share"]
    assert medium["mean_cycle_minutes"] < small["mean_cycle_minutes"]
    assert large["mean_cycle_minutes"] > medium["mean_cycle_minutes"]


def test_scenario_validation():
    with pytest.raises(ValueError, match="positive"):
        simulate(Scenario(0, 0.05))
    with pytest.raises(ValueError, match="sum to one"):
        simulate(Scenario(4, 0.05, (0.5, 0.5, 0.5)))
