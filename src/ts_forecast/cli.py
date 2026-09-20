"""Command-line interface for the time series forecasting toolkit."""

import argparse

from ts_forecast.evaluation import compute_metrics, seasonal_naive_drift_diagnostic
from ts_forecast.models import (
    forecast_arima,
    holt_winters_forecast,
    seasonal_naive_drift_forecast,
)
from ts_forecast.preprocessing import load_csv, train_test_split


def _print_metrics(title, metrics):
    print(f"{title}:")
    for key, value in metrics.items():
        if value is None:
            print(f"  {key}: n/a")
        else:
            print(f"  {key}: {value:.4f}")


def _print_diagnostic(report):
    print("Seasonal-naive + drift ensemble diagnostic")
    print(
        f"  weights: seasonal_naive={report['weights']['seasonal_naive']:.4f} "
        f"drift={report['weights']['drift']:.4f}"
    )
    print(f"  preferred: {report['preferred']}")
    for name in ("seasonal_naive", "drift", "ensemble"):
        metrics = report["metrics"][name]
        print(
            f"  {name:<16} MAE={metrics['mae']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}"
        )
    for label, value in report["skill"].items():
        if value is None:
            print(f"  {label}: n/a")
        else:
            print(f"  {label}: {value:.2f}%")


def build_parser():
    parser = argparse.ArgumentParser(description="Time Series Forecasting CLI")
    parser.add_argument("data", help="Path to CSV file")
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--steps", type=int, default=7, help="Forecast horizon")
    parser.add_argument("--order", default="1,1,1", help="ARIMA order (p,d,q)")
    parser.add_argument(
        "--model",
        default="arima",
        choices=("arima", "holt_winters", "seasonal_naive_drift"),
        help="Forecast model. Holt-Winters is ETS; seasonal_naive_drift "
        "blends seasonal-naive with random-walk-with-drift.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Season length for Holt-Winters and the seasonal-naive + drift ensemble",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Score seasonal-naive, drift, and their ensemble with kit metrics",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="equal",
        choices=("equal", "inverse_mae"),
        help="How to blend seasonal-naive and drift when using that model or --diagnose",
    )
    return parser


def run_cli(args):
    df = load_csv(args.data, args.date_col, args.target)
    train, test = train_test_split(df, args.target, test_size=args.test_size)
    steps = min(int(args.steps), len(test))
    actual = test[args.target].values[:steps]
    weights = None if args.ensemble_weights == "equal" else args.ensemble_weights

    if args.diagnose:
        report = seasonal_naive_drift_diagnostic(
            train,
            args.target,
            actual,
            steps=steps,
            seasonal_period=args.seasonal_period,
            weights=weights,
        )
        _print_diagnostic(report)
        return report

    if args.model == "arima":
        order = tuple(map(int, args.order.split(",")))
        forecast = forecast_arima(train, args.target, order=order, steps=steps)
        title = f"ARIMA{order} forecast metrics"
    elif args.model == "holt_winters":
        forecast = holt_winters_forecast(
            train,
            args.target,
            steps=steps,
            seasonal_period=args.seasonal_period,
            trend="add",
            seasonal="add",
        )
        title = "Holt-Winters forecast metrics"
    else:
        forecast = seasonal_naive_drift_forecast(
            train,
            args.target,
            steps=steps,
            seasonal_period=args.seasonal_period,
            weights=weights,
        )
        title = "Seasonal-naive + drift ensemble metrics"

    metrics = compute_metrics(actual, forecast)
    _print_metrics(title, metrics)
    return {"forecast": forecast, "metrics": metrics}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
