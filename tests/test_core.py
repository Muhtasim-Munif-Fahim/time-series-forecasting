"""Tests for time series forecasting toolkit."""

import pandas as pd
import numpy as np
import pytest
from ts_forecast.preprocessing import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    train_test_split,
    drop_na_features,
)
from ts_forecast.evaluation import (
    compute_metrics,
    conformal_prediction_interval,
    forecast_bias,
)
from ts_forecast.models import seasonal_naive_forecast


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"value": np.random.randn(100).cumsum()}, index=dates)
    return df


def test_add_lag_features(sample_df):
    result = add_lag_features(sample_df, ["value"], lags=(1, 2))
    assert "value_lag_1" in result.columns
    assert "value_lag_2" in result.columns
    assert result["value_lag_1"].iloc[1] == sample_df["value"].iloc[0]


def test_add_rolling_features(sample_df):
    result = add_rolling_features(sample_df, ["value"], windows=(7,))
    assert "value_rolling_mean_7" in result.columns
    assert "value_rolling_std_7" in result.columns


def test_add_calendar_features(sample_df):
    result = add_calendar_features(sample_df)
    assert "dayofweek" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns


def test_train_test_split(sample_df):
    train, test = train_test_split(sample_df, "value", test_size=0.2)
    assert len(train) + len(test) == len(sample_df)
    assert len(test) > 0


def test_drop_na_features(sample_df):
    df = add_lag_features(sample_df, ["value"], lags=(1, 2))
    assert df.isna().any().any()
    clean = drop_na_features(df)
    assert not clean.isna().any().any()
    assert len(clean) < len(df)


def test_compute_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.0, 2.9, 4.2, 5.0])
    metrics = compute_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_forecast_bias():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.2, 2.2, 3.2])
    bias = forecast_bias(y_true, y_pred)
    assert pytest.approx(bias) == 0.2


def test_seasonal_naive_repeats_the_latest_season():
    frame = pd.DataFrame({"value": [1, 2, 3, 10, 20, 30]})
    prediction = seasonal_naive_forecast(
        frame, "value", steps=5, seasonal_period=3
    )
    assert prediction.tolist() == [10.0, 20.0, 30.0, 10.0, 20.0]


def test_seasonal_naive_requires_a_complete_season():
    frame = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(ValueError, match="complete seasonal period"):
        seasonal_naive_forecast(frame, "value", seasonal_period=3)


def test_conformal_interval_uses_finite_sample_residual_quantile():
    lower, upper = conformal_prediction_interval(
        calibration_true=[10, 20, 30, 40],
        calibration_pred=[9, 18, 33, 40],
        forecast=[50, 60],
        coverage=0.8,
    )
    assert lower.tolist() == [47.0, 57.0]
    assert upper.tolist() == [53.0, 63.0]


def test_conformal_interval_ignores_nonfinite_calibration_pairs():
    lower, upper = conformal_prediction_interval(
        calibration_true=[1, np.nan, 3],
        calibration_pred=[0, 2, 3],
        forecast=[5],
        coverage=0.5,
    )
    assert lower.tolist() == [4.0]
    assert upper.tolist() == [6.0]


def test_conformal_interval_validates_coverage():
    with pytest.raises(ValueError, match="strictly between"):
        conformal_prediction_interval([1], [1], [2], coverage=1.0)
