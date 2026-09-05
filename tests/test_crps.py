"""Tests for the continuous ranked probability score."""

import numpy as np
import pytest

from ts_forecast.evaluation import (
    continuous_ranked_probability_score as crps,
)


def test_crps_reduces_to_mean_absolute_error_for_point_forecasts():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 5.0])
    assert crps(observed, predicted) == pytest.approx(np.mean(np.abs(observed - predicted)))


def test_crps_is_zero_for_perfect_point_forecasts():
    observed = np.array([4.0, 7.0, 1.0, 9.0])
    assert crps(observed, observed.copy()) == 0.0


def test_known_two_member_ensemble_value():
    observed = np.array([0.0, 10.0])
    forecasts = np.array([[-1.0, 1.0], [9.0, 11.0]])
    # each row: per-obs mean abs error 1, pairwise mean abs diff 1 -> 1 - 0.5 = 0.5
    assert crps(observed, forecasts) == pytest.approx(0.5)


def test_perfect_ensemble_scores_lower_than_scattered_one():
    observed = np.zeros(5)
    perfect = np.full((5, 4), 0.0)
    scattered = np.array([[-2.0, -1.0, 1.0, 2.0]] * 5)
    assert crps(observed, perfect) < crps(observed, scattered)


def test_crps_sharper_well_calibrated_ensemble_beats_wider_one():
    observed = np.array([0.0, 0.0, 0.0])
    sharp = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    wide = np.array([[-10.0, 0.0, 10.0], [-10.0, 0.0, 10.0], [-10.0, 0.0, 10.0]])
    assert crps(observed, sharp) < crps(observed, wide)


def test_crps_single_member_matches_absolute_error():
    observed = np.array([3.0, -1.0])
    single = np.array([[3.0], [-5.0]])
    assert crps(observed, single) == pytest.approx(np.mean([0.0, 4.0]))


def test_crps_validates_shapes():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        crps([1.0, 2.0], np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="equal length"):
        crps([1.0, 2.0, 3.0], np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_crps_validates_finiteness():
    with pytest.raises(ValueError, match="finite"):
        crps([1.0, np.nan], np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="finite"):
        crps([1.0, 2.0], np.array([[1.0, np.inf], [3.0, 4.0]]))


def test_crps_rejects_empty_member_axis():
    with pytest.raises(ValueError, match="at least one member"):
        crps([1.0, 2.0], np.empty((2, 0)))
