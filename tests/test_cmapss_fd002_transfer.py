"""FD002 boundary validation and frozen-input guard."""

import numpy as np
import pandas as pd
import pytest

from fabstack import cmapss_fd002_transfer


def _frame(lengths: list[int]) -> pd.DataFrame:
    values = np.zeros((sum(lengths), 26))
    values[:, 0] = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    values[:, 1] = np.concatenate([np.arange(1, length + 1) for length in lengths])
    return pd.DataFrame(values)


def test_fd002_validates_unknown_engine_counts_and_label_alignment():
    train, test, rul = _frame([3, 4]), _frame([2, 5, 3]), pd.DataFrame([10, 20, 30])
    checked_train, checked_test, labels = cmapss_fd002_transfer.validate_fd002(
        train.copy(), test.copy(), rul)
    assert (len(checked_train), len(checked_test), labels.tolist()) == (7, 10, [10, 20, 30])
    with pytest.raises(ValueError, match="label count"):
        cmapss_fd002_transfer.validate_fd002(train.copy(), test.copy(), rul.iloc[:-1])
    wrong_id = test.copy()
    wrong_id.loc[wrong_id.iloc[:, 0] == 3, 0] = 4
    with pytest.raises(ValueError, match="unit IDs"):
        cmapss_fd002_transfer.validate_fd002(train.copy(), wrong_id, rul)
    wrong_cycle = test.copy()
    wrong_cycle.iloc[1, 1] = 4
    with pytest.raises(ValueError, match="cycles"):
        cmapss_fd002_transfer.validate_fd002(train.copy(), wrong_cycle, rul)
    nonfinite = test.copy()
    nonfinite.iloc[0, 4] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        cmapss_fd002_transfer.validate_fd002(train.copy(), nonfinite, rul)


def test_fd002_guard_rejects_protocol_drift_before_data_access(tmp_path, monkeypatch):
    changed = tmp_path / "protocol.md"
    changed.write_text("changed protocol")
    monkeypatch.setattr(cmapss_fd002_transfer, "PROTOCOL", changed)
    with pytest.raises(RuntimeError, match="frozen FD002 protocol changed"):
        cmapss_fd002_transfer.guard()
