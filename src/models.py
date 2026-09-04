"""Forecasting models: baselines, statistical, and ML approaches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def moving_average_forecast(series: pd.Series, window: int = 7, horizon: int = 10) -> np.ndarray:
    last_window = series.iloc[-window:].mean()
    return np.full(horizon, last_window)


def naive_forecast(series: pd.Series, horizon: int = 10) -> np.ndarray:
    return np.full(horizon, series.iloc[-1])


def linear_regression_forecast(X_train, y_train, X_test) -> tuple[np.ndarray, LinearRegression]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def random_forest_forecast(X_train, y_train, X_test, n_estimators: int = 200) -> tuple[np.ndarray, RandomForestRegressor]:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def gradient_boosting_forecast(X_train, y_train, X_test) -> tuple[np.ndarray, GradientBoostingRegressor]:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mape": round(float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100), 4),
    }


def forecast_arima(df: pd.DataFrame, target_col: str, order=(1, 1, 1), steps=1) -> np.ndarray:
    """Forecast with an ARIMA(p, d, q) model fit via statsmodels.

    Returns a numpy array of length ``steps`` so the output is consistent
    with every other forecast function in this module. ``order`` accepts the
    standard ``(p, d, q)`` tuple or a ``(p, d, q, P, D, Q, s)`` tuple for
    seasonal ARIMA; when seasonal terms are supplied they are forwarded to
    :class:`~statsmodels.tsa.statespace.sarimax.SARIMAX`.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if target_col not in df:
        raise KeyError(f"unknown target column: {target_col}")

    series = df[target_col].dropna()
    values = np.asarray(series, dtype=float)
    if values.size < 2:
        raise ValueError("training data must contain at least two observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("training data must contain only finite values")

    if len(order) == 7:
        p, d, q, P, D, Q, s = order
        model = SARIMAX(
            values,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        model = ARIMA(values, order=order)

    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return np.asarray(forecast, dtype=float)

