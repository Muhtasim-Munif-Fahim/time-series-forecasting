"""Command-line interface for the time series forecasting toolkit."""

import argparse
import pandas as pd
from ts_forecast.preprocessing import load_csv, train_test_split, add_lag_features, add_rolling_features, add_calendar_features, drop_na_features
from ts_forecast.models import forecast_arima
from ts_forecast.evaluation import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Time Series Forecasting CLI")
    parser.add_argument("data", help="Path to CSV file")
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--steps", type=int, default=7, help="Forecast horizon")
    parser.add_argument("--order", default="1,1,1", help="ARIMA order (p,d,q)")
    args = parser.parse_args()

    df = load_csv(args.data, args.date_col, args.target)
    train, test = train_test_split(df, args.target, test_size=args.test_size)

    order = tuple(map(int, args.order.split(",")))
    forecast = forecast_arima(train, args.target, order=order, steps=args.steps)

    actual = test[args.target].values[:args.steps]
    metrics = compute_metrics(actual, forecast.values)
    print(f"ARIMA{order} forecast metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
