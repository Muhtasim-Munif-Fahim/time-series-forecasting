"""Main pipeline: end-to-end time series forecasting."""

from __future__ import annotations

import json
from pathlib import Path

from data_generation import generate_time_series
from preprocessing import create_supervised_data, temporal_split, add_datetime_features, add_rolling_features
from models import (
    linear_regression_forecast, random_forest_forecast,
    gradient_boosting_forecast, compute_metrics,
    moving_average_forecast, naive_forecast,
)


def run_pipeline(output_dir: str | Path = "output", n_points: int = 500) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = generate_time_series(n_points=n_points)
    df = add_datetime_features(df)
    df = add_rolling_features(df, "value", windows=[7, 14])
    train, test = temporal_split(df, train_ratio=0.8)

    X, y = create_supervised_data(df, target="value", lags=[1, 2, 3, 7])
    X_train, X_test = X.iloc[: len(train)], X.iloc[len(train):]
    y_train, y_test = y.iloc[: len(train)], y.iloc[len(train):]

    results = {}

    lr_preds, _ = linear_regression_forecast(X_train, y_train, X_test)
    results["linear_regression"] = compute_metrics(y_test.values, lr_preds)

    rf_preds, _ = random_forest_forecast(X_train, y_train, X_test)
    results["random_forest"] = compute_metrics(y_test.values, rf_preds)

    gb_preds, _ = gradient_boosting_forecast(X_train, y_train, X_test)
    results["gradient_boosting"] = compute_metrics(y_test.values, gb_preds)

    horizon = len(test)
    ma_preds = moving_average_forecast(train["value"], window=7, horizon=horizon)
    results["moving_average"] = compute_metrics(test["value"].values, ma_preds)

    naive_preds = naive_forecast(train["value"], horizon=horizon)
    results["naive"] = compute_metrics(test["value"].values, naive_preds)

    summary = {"n_points": n_points, "n_train": len(train), "n_test": len(test), "results": results}
    (output / "results.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(output / "time_series.csv", index=False)
    return summary
