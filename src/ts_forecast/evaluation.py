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


def compare_models(results):
    comparisons = []
    for name, (y_true, y_pred) in results.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        comparisons.append(metrics)
    return pd.DataFrame(comparisons).set_index("model")
