"""Model selection and hyperparameter tuning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Mapping

from ts_forecast.models import forecast_arima
from ts_forecast.evaluation import compute_metrics
from ts_forecast.models import rolling_origin_backtest


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


def select_model_by_backtest(
    model_fns: Mapping[str, Callable],
    df: pd.DataFrame,
    target_col: str,
    *,
    initial_window: int,
    horizon: int = 1,
    step: int = 1,
    window: int | None = None,
    gap: int = 0,
    metric: str = "rmse",
) -> tuple[str, pd.DataFrame]:
    """Rank named forecasting callables using the same rolling-origin folds.

    Returns the winning model name and a score table containing the requested
    metric plus MAE, RMSE, MAPE, fold count, and prediction count. Ties are
    resolved by model name for reproducible selection.
    """

    allowed_metrics = {"mae", "rmse", "mape"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of: {', '.join(sorted(allowed_metrics))}")
    if not model_fns:
        raise ValueError("model_fns must contain at least one model")

    rows = []
    for name, model_fn in model_fns.items():
        if not callable(model_fn):
            raise TypeError(f"model '{name}' is not callable")
        result = rolling_origin_backtest(
            model_fn,
            df,
            target_col,
            initial_window=initial_window,
            horizon=horizon,
            step=step,
            window=window,
            gap=gap,
        )
        metrics = compute_metrics(result["actual"], result["prediction"])
        rows.append(
            {
                "model": str(name),
                "score": float(metrics[metric]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "folds": int(result["fold"].nunique()),
                "predictions": int(len(result)),
            }
        )

    scores = pd.DataFrame(rows).sort_values(["score", "model"]).set_index("model")
    return str(scores.index[0]), scores
