"""Forecasting models: baselines, statistical, and ML approaches."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
