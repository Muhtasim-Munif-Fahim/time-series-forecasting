"""Time series forecasting models."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


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
