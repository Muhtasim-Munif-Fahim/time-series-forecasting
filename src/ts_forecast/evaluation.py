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


def residual_autocorrelation(y_true, y_pred, max_lag=1):
    """Measure remaining serial correlation in one-step forecast errors.

    A lag-one autocorrelation close to zero means the errors look like noise;
    strong positive correlation signals that the model still leaves
    exploitable structure on the table. Returns per-lag correlation values.
    """

    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")
    if observed.size <= max_lag + 1:
        raise ValueError("need more observations than max_lag to estimate autocorrelation")

    residual = observed - predicted
    centered = residual - residual.mean()
    denominator = np.dot(centered, centered)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("residuals must have non-zero variance")

    lags = np.arange(1, max_lag + 1)
    values = []
    for lag in lags:
        numerator = np.dot(centered[lag:], centered[:-lag])
        values.append(float(numerator / denominator))
    return {f"lag_{int(lag)}": value for lag, value in zip(lags, values)}


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


def root_mean_squared_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
    """Return RMSSE against an in-sample seasonal-naive benchmark.

    RMSSE penalises large forecast misses more heavily than MASE while still
    remaining comparable across series with different scales.
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
    scale = np.mean((training[seasonal_period:] - training[:-seasonal_period]) ** 2)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("seasonal-naive training error must be finite and non-zero")
    return float(np.sqrt(np.mean((observed - predicted) ** 2) / scale))


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


def interval_calibration_curve(y_true, intervals):
    """Compare nominal and empirical coverage across multiple interval levels.

    ``intervals`` maps each nominal coverage level to a ``(lower, upper)``
    pair. The returned table is sorted by nominal coverage for direct plotting.
    """

    if not isinstance(intervals, dict) or not intervals:
        raise ValueError("intervals must be a non-empty mapping of coverage levels")
    rows = []
    for nominal, bounds in intervals.items():
        try:
            coverage = float(nominal)
            lower, upper = bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("each interval must be a (lower, upper) pair") from exc
        metrics = interval_metrics(y_true, lower, upper, coverage=coverage)
        rows.append(
            {
                "nominal_coverage": coverage,
                "empirical_coverage": metrics["coverage"],
                "mean_width": metrics["mean_width"],
            }
        )
    return pd.DataFrame(rows).sort_values("nominal_coverage").reset_index(drop=True)


def quantile_loss(y_true, forecasts, quantiles=None):
    """Score quantile forecasts with the pinball loss.

    ``forecasts`` may be a 2-D array (rows = horizons, columns = quantiles)
    or a mapping from quantile to a 1-D forecast. The pinball loss is the
    standard proper scoring rule for probabilistic point forecasts: it is
    asymmetric, penalising under-forecasts more at high quantiles and
    over-forecasts more at low quantiles. Lower is better.
    """

    observed = np.asarray(y_true, dtype=float).ravel()
    if quantiles is None:
        quantiles = (0.1, 0.5, 0.9)

    if isinstance(forecasts, dict):
        labels = sorted(forecasts)
        matrix = np.column_stack([np.asarray(forecasts[q], dtype=float).ravel() for q in labels])
        effective_quantiles = [float(q) for q in labels]
    else:
        matrix = np.asarray(forecasts, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("forecasts must be 2-D (horizons x quantiles) or a dict")
        if matrix.shape[1] != len(quantiles):
            raise ValueError(
                f"expected {len(quantiles)} quantile columns, got {matrix.shape[1]}"
            )
        effective_quantiles = [float(q) for q in quantiles]

    if not all(0.0 < q < 1.0 for q in effective_quantiles):
        raise ValueError("quantiles must be strictly between 0 and 1")
    if matrix.shape[0] != observed.size:
        raise ValueError("y_true and forecasts must have equal length")
    if observed.size == 0:
        raise ValueError("at least one forecast is required")

    residuals = observed[:, None] - matrix
    losses = np.where(
        residuals >= 0,
        np.asarray(effective_quantiles) * residuals,
        (np.asarray(effective_quantiles) - 1.0) * residuals,
    )
    mean_loss = float(np.mean(losses))
    per_quantile = {
        f"q{int(round(q * 100)):02d}": float(losses[:, i].mean())
        for i, q in enumerate(effective_quantiles)
    }
    return {"pinball_loss": mean_loss, "per_quantile": per_quantile}


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


def summarize_interval_backtest(results, coverage=0.9):
    """Summarize rolling-origin interval coverage separately by horizon.

    ``results`` must contain ``horizon``, ``actual``, ``lower``, and ``upper``
    columns. The output combines count, empirical coverage, mean width, and
    Winkler score so interval degradation at longer horizons is visible.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas.DataFrame")
    required = {"horizon", "actual", "lower", "upper"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            f"interval backtest results missing columns: {', '.join(missing)}"
        )
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    columns = ["count", "coverage", "mean_width", "winkler_score"]
    if results.empty:
        return pd.DataFrame(columns=columns).rename_axis("horizon")

    rows = []
    for horizon, group in results.groupby("horizon", sort=True):
        metrics = interval_metrics(
            group["actual"], group["lower"], group["upper"], coverage=coverage
        )
        rows.append(
            {
                "horizon": horizon,
                "count": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")[columns]


def compare_models(results):
    comparisons = []
    for name, (y_true, y_pred) in results.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        comparisons.append(metrics)
    return pd.DataFrame(comparisons).set_index("model")
