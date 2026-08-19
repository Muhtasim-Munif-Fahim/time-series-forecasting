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
    interval_metrics,
    mean_absolute_scaled_error,
    summarize_backtest,
)
from ts_forecast.models import ensemble_forecast, rolling_origin_backtest, seasonal_naive_forecast


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


def test_mean_absolute_scaled_error_uses_seasonal_naive_scale():
    score = mean_absolute_scaled_error(
        [11.0, 13.0], [10.0, 15.0], [2.0, 4.0, 6.0, 8.0]
    )
    assert score == pytest.approx(0.75)


def test_mean_absolute_scaled_error_rejects_constant_baseline():
    with pytest.raises(ValueError, match="non-zero"):
        mean_absolute_scaled_error([2.0], [1.0], [3.0, 3.0])


def test_interval_metrics_penalize_missed_intervals():
    metrics = interval_metrics(
        y_true=[1.0, 5.0], lower=[0.0, 2.0], upper=[2.0, 4.0], coverage=0.8
    )
    assert metrics["coverage"] == 0.5
    assert metrics["mean_width"] == 2.0
    assert metrics["winkler_score"] == pytest.approx(7.0)


def test_interval_metrics_reject_inverted_bounds():
    with pytest.raises(ValueError, match="must not exceed"):
        interval_metrics([1.0], [2.0], [0.0])


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


def test_ensemble_forecast_normalizes_model_weights():
    result = ensemble_forecast([[10.0, 20.0], [14.0, 10.0]], weights=[3, 1])
    assert result.tolist() == [11.0, 17.5]


def test_ensemble_forecast_validates_horizons_and_weights():
    with pytest.raises(ValueError, match="same horizon"):
        ensemble_forecast([[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_forecast([[1.0], [2.0]], weights=[1, -1])


def test_rolling_origin_backtest_returns_tidy_horizon_rows():
    frame = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5, 6]},
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    def repeat_last(train, target_col, steps):
        return np.repeat(train[target_col].iloc[-1], steps)

    result = rolling_origin_backtest(
        repeat_last, frame, "value", initial_window=3, horizon=2, step=1
    )
    assert result[["fold", "horizon"]].values.tolist() == [
        [1, 1], [1, 2], [2, 1], [2, 2]
    ]
    assert result["prediction"].tolist() == [3.0, 3.0, 4.0, 4.0]
    assert result["actual"].tolist() == [4.0, 5.0, 5.0, 6.0]


def test_rolling_origin_backtest_validates_prediction_length():
    frame = pd.DataFrame({"value": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="exactly horizon"):
        rolling_origin_backtest(
            lambda train, target_col, steps: [1],
            frame,
            "value",
            initial_window=2,
            horizon=2,
        )


def test_rolling_origin_backtest_gap_excludes_recent_observations():
    frame = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5, 6, 7]},
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )
    seen = []

    def remember_last(train, target_col, steps):
        seen.append(train[target_col].tolist())
        return [train[target_col].iloc[-1]] * steps

    result = rolling_origin_backtest(
        remember_last,
        frame,
        "value",
        initial_window=3,
        gap=2,
        horizon=1,
        step=1,
    )

    assert seen[0] == [1, 2, 3]
    assert result["actual"].tolist() == [6, 7]
    assert result["origin"].iloc[0] == frame.index[2]


def test_rolling_origin_backtest_rejects_negative_gap():
    with pytest.raises(ValueError, match="non-negative"):
        rolling_origin_backtest(
            lambda train, target_col, steps: [1],
            pd.DataFrame({"value": [1, 2, 3]}),
            "value",
            initial_window=1,
            gap=-1,
        )


def test_summarize_backtest_reports_horizon_specific_error():
    results = pd.DataFrame(
        {
            "horizon": [1, 2, 1, 2],
            "actual": [10.0, 20.0, 12.0, 22.0],
            "prediction": [11.0, 18.0, 11.0, 25.0],
        }
    )
    summary = summarize_backtest(results)
    assert summary.loc[1, "count"] == 2
    assert summary.loc[1, "mae"] == 1.0
    assert summary.loc[1, "bias"] == 0.0
    assert summary.loc[2, "rmse"] == pytest.approx(np.sqrt(6.5))


def test_summarize_backtest_validates_tidy_schema():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_backtest(pd.DataFrame({"actual": [1.0]}))


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
