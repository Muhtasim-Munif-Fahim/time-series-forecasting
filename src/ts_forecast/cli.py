"""Command-line interface for the time series forecasting toolkit."""

import argparse

from ts_forecast.diagnostics import (
    detect_seasonal_periods,
    detrended_fluctuation_analysis,
    variance_ratio_test,
)
from ts_forecast.evaluation import compute_metrics, seasonal_naive_drift_diagnostic
from ts_forecast.nonlinearity import nonlinearity_report
from ts_forecast.trend import trend_report
from ts_forecast.models import (
    adida_forecast,
    classify_demand,
    imapa_forecast,
    croston_forecast,
    tsb_forecast,
    forecast_arima,
    holt_winters_forecast,
    fourier_regression_forecast,
    sarima_forecast,
    seasonal_naive_drift_forecast,
    structural_prediction_interval,
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
        choices=(
            "arima",
            "holt_winters",
            "fourier_regression",
            "seasonal_naive_drift",
            "sarima",
            "croston",
            "tsb",
            "adida",
            "imapa",
            "local_level",
            "local_linear_trend",
            "ssa",
        ),
        help="Forecast model. Holt-Winters is ETS; seasonal_naive_drift "
        "blends seasonal-naive with random-walk-with-drift; sarima is the "
        "fixed SARIMA(1,1,1)(1,0,1)s diagnostic; croston smooths intermittent "
        "demand size and inter-demand interval separately; tsb is the "
        "Teunter-Syntetos-Babai probability smoother for intermittent demand; "
        "adida aggregates into buckets, forecasts the totals, and spreads "
        "them back over the periods; imapa averages ADIDA over every "
        "aggregation level from 1 to the mean inter-demand interval; "
        "local_level and local_linear_trend are Kalman-filter structural "
        "models with MLE variances and exact Gaussian intervals; ssa is a "
        "Singular Spectrum Analysis recurrent forecast.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Season length for Holt-Winters, SARIMA(1,1,1)(1,0,1)s, "
        "and the seasonal-naive + drift ensemble",
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
    parser.add_argument(
        "--croston-alpha",
        type=float,
        default=0.1,
        help="Shared Croston smoothing constant for demand size and interval",
    )
    parser.add_argument(
        "--croston-alpha-size",
        type=float,
        default=None,
        help="Demand-size smoothing constant; overrides --croston-alpha",
    )
    parser.add_argument(
        "--croston-alpha-interval",
        type=float,
        default=None,
        help="Inter-demand interval smoothing constant; overrides --croston-alpha",
    )
    parser.add_argument(
        "--croston-method",
        default="croston",
        choices=("croston", "sba"),
        help="Croston ratio or the Syntetos-Boylan bias correction",
    )
    parser.add_argument(
        "--tsb-alpha",
        type=float,
        default=0.1,
        help="Shared TSB smoothing constant for probability and demand size",
    )
    parser.add_argument(
        "--tsb-alpha-probability",
        type=float,
        default=None,
        help="Demand-probability smoothing constant; overrides --tsb-alpha",
    )
    parser.add_argument(
        "--tsb-alpha-demand",
        type=float,
        default=None,
        help="Demand-size smoothing constant; overrides --tsb-alpha",
    )
    parser.add_argument(
        "--adida-level",
        type=int,
        default=None,
        help="ADIDA bucket size in periods (default: rounded mean inter-demand interval)",
    )
    parser.add_argument(
        "--adida-base",
        default="ses",
        choices=("ses", "croston", "sba", "tsb"),
        help="Method used to forecast the ADIDA bucket totals",
    )
    parser.add_argument(
        "--adida-alpha",
        type=float,
        default=None,
        help="Smoothing constant for the ADIDA base method "
        "(default: optimized for ses, 0.1 otherwise)",
    )
    parser.add_argument(
        "--imapa-max-level",
        type=int,
        default=None,
        help="Largest IMAPA aggregation level (default: max(2, ceil(ADI)))",
    )
    parser.add_argument(
        "--imapa-base",
        default="sba",
        choices=("ses", "croston", "sba", "tsb", "auto"),
        help="Per-level IMAPA method; auto picks Croston/SBA by SBC class",
    )
    parser.add_argument(
        "--imapa-combine",
        default="mean",
        choices=("mean", "median"),
        help="How IMAPA combines the per-level forecasts",
    )
    parser.add_argument(
        "--ssa-window",
        type=int,
        default=None,
        help="SSA embedding window L (default: half the training length)",
    )
    parser.add_argument(
        "--ssa-rank",
        type=int,
        default=None,
        help="SSA signal components kept (default: smallest rank explaining 90%% of sum s^2)",
    )
    parser.add_argument(
        "--interval-coverage",
        type=float,
        default=0.9,
        help="Prediction-interval coverage for the structural models",
    )
    parser.add_argument(
        "--detect-seasonality",
        action="store_true",
        help="Print periodogram cycles that pass Fisher's g test on the training split and exit",
    )
    parser.add_argument(
        "--seasonality-alpha",
        type=float,
        default=0.05,
        help="Significance level for --detect-seasonality",
    )
    parser.add_argument(
        "--random-walk-test",
        action="store_true",
        help=(
            "Print Lo-MacKinlay variance ratios (with the Chow-Denning joint test) "
            "and the DFA scaling exponent for the training split and exit"
        ),
    )
    parser.add_argument(
        "--vr-periods",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Aggregation periods q for --random-walk-test",
    )
    parser.add_argument(
        "--nonlinearity-test",
        action="store_true",
        help=(
            "Fit ARIMA(--order) to the training split and test its residuals for "
            "nonlinear dependence (BDS) and ARCH effects (Engle LM, McLeod-Li), then exit"
        ),
    )
    parser.add_argument(
        "--bds-max-dim",
        type=int,
        default=3,
        help="Largest BDS embedding dimension for --nonlinearity-test",
    )
    parser.add_argument(
        "--bds-permutations",
        type=int,
        default=0,
        help="Permutation replicates for exact-size BDS p-values (0 = asymptotic only)",
    )
    parser.add_argument(
        "--trend-test",
        action="store_true",
        help=(
            "Run Mann-Kendall, Sen's slope and Pettitt's change-point test on the "
            "training split, then exit"
        ),
    )
    parser.add_argument(
        "--trend-period",
        type=int,
        default=None,
        help="Season length for the seasonal Kendall test in --trend-test (omit for none)",
    )
    parser.add_argument(
        "--classify-demand",
        action="store_true",
        help="Print the Syntetos-Boylan-Croston demand class (ADI / CV^2) and exit",
    )
    return parser


def run_cli(args):
    df = load_csv(args.data, args.date_col, args.target)
    train, test = train_test_split(df, args.target, test_size=args.test_size)
    steps = min(int(args.steps), len(test))
    actual = test[args.target].values[:steps]
    weights = None if args.ensemble_weights == "equal" else args.ensemble_weights

    if args.detect_seasonality:
        report = detect_seasonal_periods(train[args.target].values, alpha=args.seasonality_alpha)
        print("Periodogram seasonality detection (Fisher's g test)")
        if not report["periods"]:
            last = report["last_test"]
            print(f"  no significant cycle (strongest p-value {last['p_value']:.4g})")
        for item in report["periods"]:
            print(
                f"  period {item['period']:.2f} (~{item['rounded_period']}): "
                f"g={item['g']:.4f}  p-value={item['p_value']:.3g}"
            )
        print(f"  suggested --seasonal-period: {report['primary_period']}")
        return report

    if args.random_walk_test:
        values = train[args.target].values
        vr = variance_ratio_test(values, periods=tuple(args.vr_periods))
        print("Lo-MacKinlay variance ratio test (H0: random walk)")
        for row in vr["periods"]:
            print(
                f"  q={row['period']}: VR={row['variance_ratio']:.4f}  "
                f"z={row['z']:.3f}  z*={row['z_robust']:.3f}  p*={row['p_value_robust']:.3g}"
            )
        joint = vr["chow_denning"]
        verdict = "reject" if vr["reject_random_walk"] else "do not reject"
        print(
            f"  Chow-Denning max|z*|={joint['statistic']:.3f} (q={joint['period']})  "
            f"p={joint['p_value']:.3g}: {verdict} random walk"
        )
        try:
            dfa = detrended_fluctuation_analysis(values)
        except ValueError as exc:
            print(f"DFA-1 exponent: n/a ({exc})")
            dfa = None
        else:
            print(
                f"DFA-1 exponent alpha={dfa['alpha']:.3f} "
                f"(R^2 {dfa['r_squared']:.3f}): {dfa['regime']}"
            )
        return {"variance_ratio": vr, "dfa": dfa}

    if args.nonlinearity_test:
        from statsmodels.tsa.arima.model import ARIMA

        order = tuple(map(int, args.order.split(",")))
        values = train[args.target].dropna().to_numpy(dtype=float)
        fitted = ARIMA(values, order=order).fit()
        residuals = fitted.resid[order[1]:]  # drop the start-up residuals of the differencing
        report = nonlinearity_report(
            residuals,
            max_dim=args.bds_max_dim,
            n_permutations=args.bds_permutations,
        )
        print(f"Nonlinear dependence tests on ARIMA{order} residuals (n={residuals.size})")
        for row in report["bds"]["dimensions"]:
            extra = ""
            if "p_value_permutation" in row:
                extra = f"  perm p={row['p_value_permutation']:.3g}"
            print(f"  BDS m={row['dimension']}: W={row['statistic']:.3f}  p={row['p_value']:.3g}{extra}")
        arch = report["arch_lm"]
        print(f"  ARCH-LM({arch['lags']}): LM={arch['lm_statistic']:.3f}  p={arch['p_value']:.3g}")
        ml = report["mcleod_li"]
        print(f"  McLeod-Li({len(ml['lags'])}): Q={ml['statistic']:.3f}  p={ml['p_value']:.3g}")
        print(f"  verdict: {report['verdict']}")
        return report

    if args.trend_test:
        values = train[args.target].dropna().to_numpy(dtype=float)
        report = trend_report(values, period=args.trend_period)
        mk = report["mann_kendall"]
        sen = report["sens_slope"]
        print(f"Trend tests (n={values.size})")
        print(f"  Mann-Kendall: tau={mk['tau']:.4f}  p={mk['p_value']:.3g}  ({mk['trend']})")
        print(
            f"  Sen slope: {sen['slope']:.4g} per step  "
            f"[{sen['ci_low']:.4g}, {sen['ci_high']:.4g}]"
        )
        if report["seasonal"] is not None:
            sk = report["seasonal"]
            print(
                f"  Seasonal Kendall (period {sk['period']}): z={sk['z']:.3f}  "
                f"p={sk['p_value']:.3g}  slope/step={sk['slope_per_step']:.4g}  ({sk['trend']})"
            )
        pt = report["pettitt"]
        print(f"  Pettitt: K={pt['statistic']:.0f} at index {pt['index']}  p={pt['p_value']:.3g}")
        print(f"  verdict: {report['summary']}")
        return report

    if args.classify_demand:
        report = classify_demand(train, args.target)
        print("Syntetos-Boylan-Croston demand classification")
        print(f"  ADI: {report['adi']:.4f}  CV^2: {report['cv2']:.4f}")
        print(f"  demands: {report['n_demands']}  zero share: {report['zero_share']:.2%}")
        print(f"  category: {report['category']}")
        print(f"  recommended: {report['recommended_method']}")
        return report

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

    elif args.model == "fourier_regression":
        forecast = fourier_regression_forecast(
            train,
            args.target,
            steps=steps,
            seasonal_period=args.seasonal_period or 7,
            n_harmonics=getattr(args, "n_harmonics", 3) or 3,
        )
        title = "Fourier regression forecast metrics"
    elif args.model == "ssa":
        from .ssa import ssa_forecast

        forecast = ssa_forecast(
            train,
            args.target,
            steps=steps,
            window=args.ssa_window,
            rank=args.ssa_rank,
        )
        title = "SSA recurrent forecast metrics"
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
    elif args.model == "sarima":
        forecast = sarima_forecast(
            train,
            args.target,
            steps=steps,
            seasonal_period=args.seasonal_period,
        )
        title = "SARIMA(1,1,1)(1,0,1)s forecast metrics"
    elif args.model == "croston":
        forecast = croston_forecast(
            train,
            args.target,
            steps=steps,
            alpha=args.croston_alpha,
            alpha_size=args.croston_alpha_size,
            alpha_interval=args.croston_alpha_interval,
            method=args.croston_method,
        )
        title = "Croston intermittent-demand forecast metrics"
    elif args.model == "tsb":
        forecast = tsb_forecast(
            train,
            args.target,
            steps=steps,
            alpha=args.tsb_alpha,
            alpha_probability=args.tsb_alpha_probability,
            alpha_demand=args.tsb_alpha_demand,
        )
        title = "TSB intermittent-demand forecast metrics"
    elif args.model == "adida":
        forecast = adida_forecast(
            train,
            args.target,
            steps=steps,
            aggregation_level=args.adida_level,
            base_method=args.adida_base,
            alpha=args.adida_alpha,
        )
        title = "ADIDA intermittent-demand forecast metrics"
    elif args.model == "imapa":
        forecast = imapa_forecast(
            train,
            args.target,
            steps=steps,
            max_level=args.imapa_max_level,
            base_method=args.imapa_base,
            combine=args.imapa_combine,
        )
        title = "IMAPA intermittent-demand forecast metrics"
    elif args.model in {"local_level", "local_linear_trend"}:
        trend = args.model == "local_linear_trend"
        interval = structural_prediction_interval(
            train,
            args.target,
            steps,
            trend=trend,
            coverage=args.interval_coverage,
        )
        metrics = compute_metrics(actual, interval["forecast"])
        name = "Local linear trend" if trend else "Local level"
        _print_metrics(f"{name} (Kalman MLE) forecast metrics", metrics)
        inside = (actual >= interval["lower"]) & (actual <= interval["upper"])
        print(
            f"  {args.interval_coverage:.0%} interval coverage: "
            f"{inside.mean():.2%} over {inside.size} steps"
        )
        return {**interval, "metrics": metrics, "coverage": float(inside.mean())}
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
