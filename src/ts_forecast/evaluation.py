"""Evaluation utilities for time series forecasting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def conformal_prediction_interval(
    calibration_true,
    calibration_pred,
    forecast,
    coverage=0.9,
):
    """Build split-conformal intervals from held-out absolute residuals.

    The finite-sample corrected quantile provides marginal coverage under the
    usual exchangeability assumption. Calibration observations with non-finite
    values are ignored.
    """

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    observed = np.asarray(calibration_true, dtype=float).ravel()
    predicted = np.asarray(calibration_pred, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("calibration_true and calibration_pred must have equal length")

    finite = np.isfinite(observed) & np.isfinite(predicted)
    scores = np.abs(observed[finite] - predicted[finite])
    if scores.size == 0:
        raise ValueError("at least one finite calibration residual is required")

    quantile_level = min(
        1.0,
        np.ceil((scores.size + 1) * coverage) / scores.size,
    )
    radius = float(np.quantile(scores, quantile_level, method="higher"))
    point_forecast = np.asarray(forecast, dtype=float)
    return point_forecast - radius, point_forecast + radius


def compute_metrics(y_true, y_pred, prefix=""):
    return {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def forecast_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
    """Return MASE using an in-sample seasonal-naive scaling error.

    Unlike percentage errors, MASE remains defined when observations are zero
    and is comparable across series. Values below one beat the corresponding
    seasonal-naive forecast on average.
    """

    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    training = np.asarray(y_train, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if training.size <= seasonal_period:
        raise ValueError("y_train must contain more than one seasonal period")
    scale = np.mean(np.abs(training[seasonal_period:] - training[:-seasonal_period]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("seasonal-naive training error must be finite and non-zero")
    return float(np.mean(np.abs(observed - predicted)) / scale)


def interval_metrics(y_true, lower, upper, coverage=0.9):
    """Evaluate prediction intervals with coverage, width, and Winkler score."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    if not (observed.shape == low.shape == high.shape):
        raise ValueError("y_true, lower, and upper must have equal length")
    if observed.size == 0:
        raise ValueError("at least one interval is required")
    if np.any(low > high):
        raise ValueError("lower bounds must not exceed upper bounds")
    if not np.all(np.isfinite(np.concatenate([observed, low, high]))):
        raise ValueError("interval inputs must contain only finite values")

    width = high - low
    alpha = 1.0 - coverage
    winkler = width.copy()
    below = observed < low
    above = observed > high
    winkler[below] += (2.0 / alpha) * (low[below] - observed[below])
    winkler[above] += (2.0 / alpha) * (observed[above] - high[above])
    return {
        "coverage": float(np.mean((observed >= low) & (observed <= high))),
        "mean_width": float(np.mean(width)),
        "winkler_score": float(np.mean(winkler)),
    }


def summarize_backtest(results):
    """Summarize tidy rolling-origin predictions for each forecast horizon."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas.DataFrame")
    required = {"horizon", "actual", "prediction"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"backtest results missing columns: {', '.join(missing)}")
    if results.empty:
        return pd.DataFrame(columns=["count", "mae", "rmse", "bias"]).rename_axis(
            "horizon"
        )

    rows = []
    for horizon, group in results.groupby("horizon", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        residual = prediction - actual
        rows.append(
            {
                "horizon": horizon,
                "count": int(len(group)),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "bias": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def compare_models(results):
    comparisons = []
    for name, (y_true, y_pred) in results.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        comparisons.append(metrics)
    return pd.DataFrame(comparisons).set_index("model")
