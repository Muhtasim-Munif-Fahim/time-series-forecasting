"""Tests for dominant-period selection in the seasonal-naive forecaster."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.models import (
    seasonal_naive_auto,
    seasonal_naive_forecast,
    select_season_length,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _sine_series(period, cycles):
    steps = np.arange(period * cycles, dtype=float)
    return 10.0 + 3.0 * np.sin(2.0 * np.pi * steps / period)


def test_select_season_length_finds_dominant_period():
    frame = _frame(_sine_series(7, cycles=60))
    assert select_season_length(frame, "value", [7, 30, 365]) == 7
    assert select_season_length(frame, "value", [365, 30, 7, 7]) == 7


def test_select_season_length_prefers_true_cycle_over_divisors():
    frame = _frame(_sine_series(12, cycles=4))
    assert select_season_length(frame, "value", [2, 3, 4, 6, 12]) == 12


def test_select_season_length_prefers_shortest_aligned_lag():
    frame = _frame(np.tile([1.0, 2.0], 20))
    assert select_season_length(frame, "value", [6, 2]) == 2


def test_select_season_length_ignores_missing_values():
    values = _sine_series(7, cycles=60)
    values[-1] = np.nan
    frame = _frame(values)
    assert select_season_length(frame, "value", [7, 13]) == 7


def test_seasonal_naive_auto_matches_manual_selection():
    frame = _frame(_sine_series(7, cycles=60))
    auto = seasonal_naive_auto(frame, "value", steps=10, candidates=[5, 7, 14])
    manual = seasonal_naive_forecast(frame, "value", steps=10, seasonal_period=7)
    assert isinstance(auto, np.ndarray)
    np.testing.assert_allclose(auto, manual)


def test_seasonal_naive_auto_repeats_selected_season():
    values = _sine_series(7, cycles=60)
    forecast = seasonal_naive_auto(
        _frame(values), "value", steps=9, candidates=[3, 7]
    )
    assert forecast.shape == (9,)
    np.testing.assert_allclose(forecast, np.resize(values[-7:], 9))


def test_select_season_length_validates_inputs():
    frame = _frame(np.arange(40.0))

    with pytest.raises(KeyError, match="unknown target column"):
        select_season_length(frame, "missing", [7])

    with pytest.raises(ValueError, match="non-empty sequence"):
        select_season_length(frame, "value", [])

    with pytest.raises(ValueError, match="non-empty sequence"):
        select_season_length(frame, "value", 7)

    with pytest.raises(ValueError, match="non-empty sequence"):
        select_season_length(frame, "value", [7.5])

    with pytest.raises(ValueError, match="non-empty sequence"):
        select_season_length(frame, "value", [True])

    with pytest.raises(ValueError, match="at least 2"):
        select_season_length(frame, "value", [1, 7])

    with pytest.raises(ValueError, match="longest candidate period"):
        select_season_length(frame, "value", [50])

    with pytest.raises(ValueError, match="only finite values"):
        gapped = _frame(np.arange(40.0))
        gapped.loc[5, "value"] = np.inf
        select_season_length(gapped, "value", [7])

    with pytest.raises(ValueError, match="at least one observation"):
        select_season_length(_frame([np.nan]), "value", [7])

    with pytest.raises(ValueError, match="non-zero variance"):
        select_season_length(_frame(np.full(40, 5.0)), "value", [7])


def test_seasonal_naive_auto_propagates_horizon_validation():
    frame = _frame(_sine_series(7, cycles=60))
    with pytest.raises(ValueError, match="steps must be at least 1"):
        seasonal_naive_auto(frame, "value", steps=0, candidates=[7])
