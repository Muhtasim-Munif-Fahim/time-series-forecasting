"""Tests for the Diebold-Mariano forecast comparison test."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.evaluation import diebold_mariano_test


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_statistic_matches_manual_newey_west_computation():
    observed = np.zeros(4)
    first = np.array([1.0, np.sqrt(3.0), 1.0, np.sqrt(3.0)])
    second = np.zeros(4)

    result = diebold_mariano_test(observed, first, second)

    assert result["dm_stat"] == pytest.approx(4.0)
    assert result["p_value"] == pytest.approx(6.334248366623973e-05)


def test_max_lag_uses_autocorrelation_of_the_differential():
    observed = np.zeros(4)
    first = np.array([1.0, np.sqrt(3.0), 1.0, np.sqrt(3.0)])
    second = np.zeros(4)

    lagged = diebold_mariano_test(observed, first, second, max_lag=1)
    corrected = diebold_mariano_test(
        observed, first, second, max_lag=1, small_sample=True
    )

    assert lagged["dm_stat"] == pytest.approx(8.0)
    assert lagged["p_value"] == pytest.approx(1.244192114854348e-15)
    assert corrected["dm_stat"] == pytest.approx(8.0 * np.sqrt(3.0 / 4.0))
    assert corrected["p_value"] == pytest.approx(0.006165373138837153)


def test_absolute_loss_option():
    observed = np.zeros(4)
    first = np.array([1.0, -2.0, 1.0, -2.0])
    second = np.zeros(4)

    result = diebold_mariano_test(observed, first, second, loss="absolute")

    assert result["dm_stat"] == pytest.approx(6.0)


def test_negative_statistic_favours_first_forecast():
    steps = np.arange(40, dtype=float)
    observed = 10.0 + 2.0 * steps
    good = observed + 0.5 * np.sin(steps / 3.0)
    biased = observed + 8.0

    result = diebold_mariano_test(observed, good, biased)

    assert result["dm_stat"] < 0
    assert result["p_value"] < 0.01


def test_swapping_forecasts_negates_the_statistic():
    rng = np.random.default_rng(7)
    observed = rng.normal(size=60)
    first = observed + rng.normal(scale=0.5, size=60)
    second = observed + rng.normal(scale=1.5, size=60)

    forward = diebold_mariano_test(observed, first, second)
    reversed_result = diebold_mariano_test(observed, second, first)

    assert forward["dm_stat"] == pytest.approx(-reversed_result["dm_stat"])
    assert forward["p_value"] == pytest.approx(reversed_result["p_value"])


def test_identical_forecasts_have_no_usable_differential():
    observed = np.arange(5.0)

    with pytest.raises(ValueError, match="finite and positive"):
        diebold_mariano_test(observed, observed.copy(), observed.copy())


def test_validates_inputs():
    observed = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="loss must be"):
        diebold_mariano_test(observed, observed, observed, loss="mse")

    with pytest.raises(ValueError, match="non-negative integer"):
        diebold_mariano_test(observed, observed, observed, max_lag=-1)

    with pytest.raises(ValueError, match="non-negative integer"):
        diebold_mariano_test(observed, observed, observed, max_lag=True)

    with pytest.raises(ValueError, match="equal length"):
        diebold_mariano_test(observed, observed, observed[:2])

    with pytest.raises(ValueError, match="at least one observation"):
        diebold_mariano_test([], [], [])

    with pytest.raises(ValueError, match="more observations than max_lag"):
        diebold_mariano_test(observed, observed, observed, max_lag=3)
