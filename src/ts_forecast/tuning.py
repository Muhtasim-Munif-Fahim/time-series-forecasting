"""Model selection and hyperparameter tuning utilities."""

import numpy as np
from ts_forecast.models import forecast_arima
from ts_forecast.evaluation import compute_metrics


def grid_search_arima(train, target_col, p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2), steps=1):
    best_score = float("inf")
    best_order = None
    results = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    forecast = forecast_arima(train, target_col, order=(p, d, q), steps=steps)
                    y_true = train[target_col].values[-steps:]
                    metrics = compute_metrics(y_true, forecast.values)
                    score = metrics["rmse"]
                    results.append({"order": (p, d, q), "rmse": score})
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                except Exception:
                    results.append({"order": (p, d, q), "rmse": np.inf})
    return best_order, results
