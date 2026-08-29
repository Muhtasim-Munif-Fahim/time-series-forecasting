"""Tests for the missing-value imputation helpers."""

import numpy as np
import pytest

from ts_forecast.preprocessing import impute_missing


def test_forward_fill_carries_last_observation():
    values = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan])
    result = impute_missing(values, method="forward")
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 4.0, 5.0, 5.0])


def test_backward_fill_carries_next_observation():
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result = impute_missing(values, method="backward")
    np.testing.assert_allclose(result, [1.0, 3.0, 3.0, 5.0, 5.0])


def test_linear_interpolation_bridges_gaps():
    values = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
    result = impute_missing(values, method="linear")
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_forward_and_backward_leave_edge_gaps_unfilled():
    values = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    forward = impute_missing(values, method="forward")
    backward = impute_missing(values, method="backward")
    assert np.isnan(forward[0])
    np.testing.assert_allclose(forward[1:], [2.0, 2.0, 4.0, 4.0])
    np.testing.assert_allclose(backward[:-1], [2.0, 2.0, 4.0, 4.0])
    assert np.isnan(backward[-1])


def test_linear_leaves_edge_gaps_unfilled():
    values = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    result = impute_missing(values, method="linear")
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:4], [2.0, 3.0, 4.0])
    assert np.isnan(result[-1])


def test_seasonal_imputation_uses_phase_means():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, np.nan, np.nan])
    result = impute_missing(values, method="seasonal", seasonal_period=3)
    np.testing.assert_allclose(result[6], 25.0)
    np.testing.assert_allclose(result[7], 35.0)


def test_seasonal_imputation_covers_leading_gaps():
    values = np.array([np.nan, np.nan, 30.0, 10.0, 20.0, 30.0])
    result = impute_missing(values, method="seasonal", seasonal_period=3)
    np.testing.assert_allclose(result[0], 10.0)
    np.testing.assert_allclose(result[1], 20.0)


def test_no_missing_values_returns_unchanged_copy():
    values = np.array([1.0, 2.0, 3.0])
    result = impute_missing(values)
    np.testing.assert_allclose(result, values)
    assert result is not values


def test_validation_errors():
    with pytest.raises(ValueError, match="method"):
        impute_missing([1.0, np.nan, 3.0], method="mean")
    with pytest.raises(ValueError, match="seasonal_period"):
        impute_missing([1.0, np.nan, 3.0], method="seasonal")
    with pytest.raises(ValueError, match="seasonal_period"):
        impute_missing([1.0, np.nan, 3.0], method="seasonal", seasonal_period=1)
    with pytest.raises(ValueError, match="only used with method"):
        impute_missing([1.0, np.nan, 3.0], method="forward", seasonal_period=7)
    with pytest.raises(ValueError, match="at least one non-missing"):
        impute_missing([np.nan, np.nan])
    with pytest.raises(ValueError, match="finite"):
        impute_missing([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="not be empty"):
        impute_missing([])
