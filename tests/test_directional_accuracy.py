"""Tests for directional accuracy of forecasted turns."""

import numpy as np
import pytest

from ts_forecast.evaluation import directional_accuracy


def test_perfect_directional_accuracy():
    observed = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    predicted = observed + 0.5
    assert directional_accuracy(observed, predicted) == 1.0


def test_all_wrong_returns_zero():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = observed[::-1]
    assert directional_accuracy(observed, predicted) == 0.0


def test_partial_accuracy_counts_each_step():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 3.0, 2.0, 4.0])
    # changes: +,+ ,+  vs  +,-,+  -> matches on 2 of 3 steps
    assert directional_accuracy(observed, predicted) == pytest.approx(2.0 / 3.0)


def test_flat_series_agrees_only_when_both_unchanged():
    observed = np.array([5.0, 5.0, 5.0, 5.0])
    assert directional_accuracy(observed, observed) == 1.0
    rising = np.array([5.0, 6.0, 7.0, 8.0])
    # every step: observed 0 vs predicted + -> disagreement
    assert directional_accuracy(observed, rising) == 0.0


def test_insensitive_to_scale_but_not_direction():
    observed = np.array([10.0, 20.0, 15.0, 30.0])
    predicted = observed * 10.0
    assert directional_accuracy(observed, predicted) == 1.0


def test_rejects_short_input():
    with pytest.raises(ValueError, match="at least two observations"):
        directional_accuracy([1.0], [2.0])


def test_rejects_mismatched_length():
    with pytest.raises(ValueError, match="equal length"):
        directional_accuracy([1.0, 2.0, 3.0], [1.0, 2.0])


def test_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        directional_accuracy([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        directional_accuracy([1.0, 2.0], [1.0, np.inf])
