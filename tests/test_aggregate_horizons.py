"""Tests for forecast horizon aggregation."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.models import aggregate_forecast_horizons


def test_aggregate_daily_to_weekly_sum():
    """Sum aggregation from daily to weekly."""
    daily_fc = np.arange(1.0, 32.0)
    daily_ts = pd.date_range('2024-01-01', periods=31, freq='D')
    weekly_fc, weekly_ts, periods = aggregate_forecast_horizons(
        daily_fc, daily_ts, freq='W', method='sum'
    )
    assert len(weekly_fc) == 5
    assert weekly_fc[0] == pytest.approx(28.0)  # 1+2+3+4+5+6+7
    assert weekly_fc[-1] == pytest.approx(90.0)  # 29+30+31
    assert all(isinstance(p, pd.Period) for p in periods)


def test_aggregate_daily_to_weekly_mean():
    """Mean aggregation from daily to weekly."""
    daily_fc = np.arange(1.0, 32.0)
    daily_ts = pd.date_range('2024-01-01', periods=31, freq='D')
    weekly_fc, _, _ = aggregate_forecast_horizons(
        daily_fc, daily_ts, freq='W', method='mean'
    )
    assert weekly_fc[0] == pytest.approx(4.0)
    assert weekly_fc[-1] == pytest.approx(30.0)


def test_aggregate_daily_to_weekly_last():
    """Last value aggregation from daily to weekly."""
    daily_fc = np.arange(1.0, 32.0)
    daily_ts = pd.date_range('2024-01-01', periods=31, freq='D')
    weekly_fc, _, _ = aggregate_forecast_horizons(
        daily_fc, daily_ts, freq='W', method='last'
    )
    assert weekly_fc[0] == 7.0
    assert weekly_fc[-1] == 31.0


def test_aggregate_daily_to_monthly():
    """Aggregate daily to monthly."""
    daily_fc = np.ones(60)
    daily_ts = pd.date_range('2024-01-01', periods=60, freq='D')
    monthly_fc, monthly_ts, periods = aggregate_forecast_horizons(
        daily_fc, daily_ts, freq='M', method='sum'
    )
    assert len(monthly_fc) == 2
    assert monthly_fc[0] == pytest.approx(31.0)
    assert monthly_fc[1] == pytest.approx(29.0)


def test_aggregate_with_level_methods():
    """Per-period method overrides."""
    daily_fc = np.arange(1.0, 32.0)
    daily_ts = pd.date_range('2024-01-01', periods=31, freq='D')
    weekly_fc, _, _ = aggregate_forecast_horizons(
        daily_fc,
        daily_ts,
        freq='W',
        method='sum',
        level_methods={'2024-01-01/2024-01-07': 'mean', '2024-01-08/2024-01-14': 'last'},
    )
    # First week uses mean, second uses last
    assert weekly_fc[0] == pytest.approx(4.0)
    assert weekly_fc[1] == 14.0
    # Remaining use default sum
    assert weekly_fc[2] == pytest.approx(126.0)


def test_aggregate_validates_method():
    """Invalid method raises error."""
    with pytest.raises(ValueError, match='method must be one of'):
        aggregate_forecast_horizons([1.0], [pd.Timestamp('2024-01-01')], method='invalid')


def test_aggregate_validates_equal_length():
    """Forecast and timestamps must have equal length."""
    with pytest.raises(ValueError, match='equal length'):
        aggregate_forecast_horizons([1.0, 2.0], [pd.Timestamp('2024-01-01')])


def test_aggregate_validates_nonempty():
    """Forecast must not be empty."""
    with pytest.raises(ValueError, match='must not be empty'):
        aggregate_forecast_horizons([], [])


def test_aggregate_validates_finite():
    """Forecast must contain only finite values."""
    with pytest.raises(ValueError, match='finite values'):
        aggregate_forecast_horizons([1.0, np.nan], pd.date_range('2024-01-01', periods=2))


def test_aggregate_returns_correct_types():
    """Return types are correct."""
    daily_fc = np.arange(1.0, 8.0)
    daily_ts = pd.date_range('2024-01-01', periods=7, freq='D')
    weekly_fc, weekly_ts, periods = aggregate_forecast_horizons(
        daily_fc, daily_ts, freq='W', method='sum'
    )
    assert isinstance(weekly_fc, np.ndarray)
    assert isinstance(weekly_ts, np.ndarray)
    assert isinstance(periods, np.ndarray)
    assert weekly_fc.dtype == float
    assert all(isinstance(t, pd.Timestamp) for t in weekly_ts)
    assert all(isinstance(p, pd.Period) for p in periods)


def test_aggregate_min_max_methods():
    """Min and max aggregation methods."""
    daily_fc = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0])
    daily_ts = pd.date_range('2024-01-01', periods=7, freq='D')
    min_fc, _, _ = aggregate_forecast_horizons(daily_fc, daily_ts, freq='W', method='min')
    max_fc, _, _ = aggregate_forecast_horizons(daily_fc, daily_ts, freq='W', method='max')
    assert min_fc[0] == 1.0
    assert max_fc[0] == 7.0


def test_aggregate_first_method():
    """First value aggregation."""
    daily_fc = np.array([10.0, 20.0, 30.0, 40.0])
    daily_ts = pd.date_range('2024-01-01', periods=4, freq='D')
    first_fc, _, _ = aggregate_forecast_horizons(daily_fc, daily_ts, freq='W', method='first')
    assert first_fc[0] == 10.0
