"""Tests for the M4 Overall Weighted Average (OWA) accuracy metric."""

import numpy as np
import pytest

from ts_forecast.evaluation import overall_weighted_average


def test_perfect_forecast_scores_zero():
    train = np.arange(1.0, 17.0)
    observed = np.array([17.0, 18.0, 19.0, 20.0])
    score = overall_weighted_average(observed, observed, train, seasonal_period=4)
    assert score == pytest.approx(0.0)


def test_seasonal_naive_achieves_owa_of_one():
    train = np.arange(1.0, 13.0)
    observed = np.array([13.0, 14.0, 15.0, 16.0])
    benchmark = np.array([9.0, 10.0, 11.0, 12.0])
    score = overall_weighted_average(observed, benchmark, train, seasonal_period=4)
    assert score == pytest.approx(1.0)


def test_worse_than_naive_exceeds_one():
    train = np.arange(10.0, 42.0, 2.0)
    observed = np.array([31.0, 33.0, 35.0, 37.0])
    forecast = np.full(4, 100.0)
    score = overall_weighted_average(observed, forecast, train, seasonal_period=4)
    assert score > 1.0


def test_better_than_naive_below_one():
    train = np.arange(10.0, 42.0, 2.0)
    observed = np.array([31.0, 33.0, 35.0, 37.0])
    forecast = np.array([30.5, 32.5, 34.5, 36.5])
    score = overall_weighted_average(observed, forecast, train, seasonal_period=4)
    assert 0.0 < score < 1.0


def test_non_seasonal_period_perfect_scores_zero():
    train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    observed = np.array([60.0, 70.0])
    score = overall_weighted_average(observed, observed, train, seasonal_period=1)
    assert score == pytest.approx(0.0)


def test_non_seasonal_naive_is_benchmark():
    train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    observed = np.array([60.0, 70.0])
    forecast = np.array([50.0, 50.0])
    score = overall_weighted_average(observed, forecast, train, seasonal_period=1)
    assert score == pytest.approx(1.0)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        overall_weighted_average([1.0, 2.0], [1.0], [1.0, 2.0, 3.0])


def test_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="finite"):
        overall_weighted_average(
            [1.0, np.nan], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]
        )


def test_rejects_insufficient_training_data():
    with pytest.raises(ValueError, match="more than one seasonal period"):
        overall_weighted_average([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], seasonal_period=2)


def test_rejects_empty_observations():
    with pytest.raises(ValueError, match="at least one observation"):
        overall_weighted_average([], [], [1.0, 2.0, 3.0, 4.0])


def test_rejects_invalid_seasonal_period():
    with pytest.raises(ValueError, match="at least 1"):
        overall_weighted_average(
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], seasonal_period=0
        )


def test_returns_float_type():
    result = overall_weighted_average(
        [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        seasonal_period=2,
    )
    assert isinstance(result, float)
