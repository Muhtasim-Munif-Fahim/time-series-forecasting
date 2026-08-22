"""Time series forecasting models."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def ensemble_forecast(forecasts, weights=None, method="mean"):
    """Blend same-horizon forecasts with mean or median aggregation.

    Parameters
    ----------
    forecasts : sequence of array-like
        Point forecasts from two or more models. Every forecast must have the
        same horizon.
    weights : sequence of float, optional
        Relative model weights. They are normalized internally, so their sum
        need not equal one.
    method : {"mean", "median"}
        Aggregation strategy. Median aggregation is robust to one anomalous
        model forecast and does not accept weights.
    """

    if forecasts is None:
        raise ValueError("at least one forecast is required")
    forecasts = list(forecasts)
    if not forecasts:
        raise ValueError("at least one forecast is required")
    arrays = [np.asarray(forecast, dtype=float).ravel() for forecast in forecasts]
    if any(array.size == 0 for array in arrays):
        raise ValueError("forecasts must not be empty")
    if len({array.size for array in arrays}) != 1:
        raise ValueError("all forecasts must have the same horizon")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("forecasts must contain only finite values")

    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")
    if method == "median":
        if weights is not None:
            raise ValueError("weights are not supported with median aggregation")
        return np.median(np.vstack(arrays), axis=0)

    if weights is None:
        normalized = np.ones(len(arrays), dtype=float)
    else:
        normalized = np.asarray(weights, dtype=float).ravel()
        if normalized.size != len(arrays):
            raise ValueError("weights must match the number of forecasts")
        if not np.all(np.isfinite(normalized)) or np.any(normalized < 0):
            raise ValueError("weights must be finite and non-negative")
        if normalized.sum() <= 0:
            raise ValueError("at least one weight must be positive")

    return np.average(np.vstack(arrays), axis=0, weights=normalized)


def seasonal_naive_forecast(train, target_col, steps=1, seasonal_period=7):
    """Repeat the most recent observed season for the requested horizon."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be at least 1")
    values = np.asarray(train[target_col].dropna(), dtype=float)
    if len(values) < seasonal_period:
        raise ValueError(
            "training data must contain at least one complete seasonal period"
        )
    season = values[-seasonal_period:]
    return np.resize(season, steps)


def forecast_arima(train, target_col, order=(1, 1, 1), steps=1):
    model = ARIMA(train[target_col].dropna(), order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return forecast


def evaluate_forecast(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def walk_forward_validation(model_fn, df, target_col, window=30, steps=1):
    predictions = []
    actuals = []
    n = len(df)
    for i in range(window, n - steps + 1):
        train_window = df.iloc[i - window : i]
        pred = model_fn(train_window, target_col, steps=steps)
        actual = df.iloc[i : i + steps][target_col].values
        predictions.append(pred)
        actuals.append(actual)
    return np.array(actuals), np.array(predictions)


def rolling_origin_backtest(
    model_fn,
    df,
    target_col,
    *,
    initial_window,
    horizon=1,
    step=1,
    window=None,
    gap=0,
    max_folds=None,
):
    """Evaluate a forecasting callable over reproducible rolling origins.

    ``model_fn`` receives ``(training_frame, target_col, steps=horizon)``.
    By default the training set expands; setting ``window`` keeps only the most
    recent observations at each origin. The tidy result preserves timestamps
    and horizon numbers for horizon-specific analysis. ``gap`` leaves the
    observations immediately before each test origin unused, which prevents
    look-ahead when a workflow has a reporting or label delay.
    """

    if target_col not in df:
        raise KeyError(f"unknown target column: {target_col}")
    if initial_window < 1 or horizon < 1 or step < 1:
        raise ValueError("initial_window, horizon, and step must be at least 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if window is not None and window < 1:
        raise ValueError("window must be at least 1 when provided")
    if max_folds is not None and (
        isinstance(max_folds, bool) or not isinstance(max_folds, int) or max_folds < 1
    ):
        raise ValueError("max_folds must be a positive integer or None")
    if initial_window + gap + horizon > len(df):
        raise ValueError("not enough observations for one backtest fold")

    records = []
    fold = 0
    first_origin = initial_window + gap
    for origin in range(first_origin, len(df) - horizon + 1, step):
        if max_folds is not None and fold >= max_folds:
            break
        train_end = origin - gap
        start = 0 if window is None else max(0, train_end - window)
        train = df.iloc[start:train_end]
        prediction = np.asarray(
            model_fn(train, target_col, steps=horizon), dtype=float
        ).ravel()
        if prediction.size != horizon:
            raise ValueError("model_fn must return exactly horizon predictions")
        actual = np.asarray(df.iloc[origin : origin + horizon][target_col], dtype=float)
        timestamps = df.index[origin : origin + horizon]
        fold += 1
        for offset, (timestamp, observed, predicted) in enumerate(
            zip(timestamps, actual, prediction), start=1
        ):
            records.append(
                {
                    "fold": fold,
                    "origin": df.index[train_end - 1],
                    "timestamp": timestamp,
                    "horizon": offset,
                    "actual": float(observed),
                    "prediction": float(predicted),
                }
            )
    return pd.DataFrame.from_records(records)
