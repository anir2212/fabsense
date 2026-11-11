"""FD004 archive boundary and frozen-input guard."""

import numpy as np
import pandas as pd
import pytest

from fabstack import cmapss, cmapss_fd004_transfer


def _frame(engines: int, total_rows: int) -> pd.DataFrame:
    base, remainder = divmod(total_rows, engines)
    lengths = [base + (index < remainder) for index in range(engines)]
    values = np.zeros((total_rows, 26))
    values[:, 0] = np.repeat(np.arange(1, engines + 1), lengths)
    values[:, 1] = np.concatenate([np.arange(1, length + 1) for length in lengths])
    return pd.DataFrame(values)


def test_fd004_requires_declared_engine_counts_and_aligned_labels():
    train, test, rul = _frame(249, 61249), _frame(248, 41214), pd.DataFrame([10] * 248)
    _, checked_test, labels = cmapss.validate_tables("FD004", train.copy(), test.copy(), rul)
    assert checked_test.shape == (41214, 26) and len(labels) == 248

    missing_engine = train.copy()
    missing_engine.loc[missing_engine.iloc[:, 0] == 249, 0] = 250
    with pytest.raises(ValueError, match="unexpected FD004 unit IDs"):
        cmapss.validate_tables("FD004", missing_engine, test.copy(), rul)
    with pytest.raises(ValueError, match="Unexpected FD004 table shape"):
        cmapss.validate_tables("FD004", train.copy(), test.copy(), rul.iloc[:-1])
    wrong_cycle = test.copy()
    wrong_cycle.iloc[1, 1] = 3
    with pytest.raises(ValueError, match="consecutive"):
        cmapss.validate_tables("FD004", train.copy(), wrong_cycle, rul)


def test_fd004_protocol_guard_rejects_changed_file_before_archive_read(tmp_path, monkeypatch):
    changed = tmp_path / "protocol.md"
    changed.write_text("changed protocol")
    monkeypatch.setattr(cmapss_fd004_transfer, "PROTOCOL", changed)
    with pytest.raises(RuntimeError, match="frozen FD004 input changed"):
        cmapss_fd004_transfer.guard()
