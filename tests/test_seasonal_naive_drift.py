"""Tests for the seasonal-naive + drift ensemble diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.evaluation import (
    forecast_accuracy,
    forecast_skill_score,
    seasonal_naive_drift_diagnostic,
)
from ts_forecast.models import (
    drift_forecast,
    ensemble_forecast,
    seasonal_naive_drift_forecast,
    seasonal_naive_forecast,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _seasonal_trend_series(cycles=8, period=4, slope=1.0):
    season = np.array([1.0, 3.0, 2.0, 4.0])
    values = []
    for cycle in range(cycles):
        values.extend(season + slope * cycle)
    return _frame(values), period


def test_drift_forecast_extrapolates_linear_slope():
    frame = _frame([0.0, 2.0, 4.0, 6.0, 8.0])
    forecast = drift_forecast(frame, "value", steps=3)
    np.testing.assert_allclose(forecast, [10.0, 12.0, 14.0])


def test_drift_forecast_is_flat_when_series_has_no_slope():
    frame = _frame([5.0, 5.0, 5.0, 5.0])
    forecast = drift_forecast(frame, "value", steps=4)
    np.testing.assert_allclose(forecast, [5.0, 5.0, 5.0, 5.0])


def test_ensemble_is_mean_of_seasonal_naive_and_drift():
    frame, period = _seasonal_trend_series()
    seasonal = seasonal_naive_forecast(frame, "value", steps=4, seasonal_period=period)
    drift = drift_forecast(frame, "value", steps=4)
    ensemble = seasonal_naive_drift_forecast(
        frame, "value", steps=4, seasonal_period=period
    )
    np.testing.assert_allclose(ensemble, ensemble_forecast([seasonal, drift]))


def test_inverse_mae_weights_tilt_toward_better_component():
    frame, period = _seasonal_trend_series()
    equal = seasonal_naive_drift_forecast(
        frame, "value", steps=4, seasonal_period=period
    )
    weighted = seasonal_naive_drift_forecast(
        frame, "value", steps=4, seasonal_period=period, weights="inverse_mae"
    )
    assert weighted.shape == equal.shape
    assert not np.allclose(weighted, equal)


def test_custom_weights_match_ensemble_forecast():
    frame, period = _seasonal_trend_series()
    seasonal = seasonal_naive_forecast(frame, "value", steps=4, seasonal_period=period)
    drift = drift_forecast(frame, "value", steps=4)
    expected = ensemble_forecast([seasonal, drift], weights=[3, 1])
    result = seasonal_naive_drift_forecast(
        frame, "value", steps=4, seasonal_period=period, weights=[3, 1]
    )
    np.testing.assert_allclose(result, expected)


def test_diagnostic_uses_forecast_accuracy_metrics():
    frame, period = _seasonal_trend_series()
    train = frame.iloc[:-4]
    holdout = frame.iloc[-4:]["value"].to_numpy()
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=period
    )

    assert set(report) == {"forecasts", "weights", "metrics", "preferred", "skill"}
    assert report["preferred"] in {"seasonal_naive", "drift", "ensemble"}
    assert report["weights"]["seasonal_naive"] == pytest.approx(0.5)
    assert report["weights"]["drift"] == pytest.approx(0.5)

    for name in ("seasonal_naive", "drift", "ensemble"):
        expected = forecast_accuracy(
            holdout,
            report["forecasts"][name],
            y_train=train["value"].to_numpy(),
            seasonal_period=period,
        )
        assert report["metrics"][name]["mae"] == pytest.approx(expected["mae"])
        assert report["metrics"][name]["rmse"] == pytest.approx(expected["rmse"])
        assert report["metrics"][name]["mape"] == pytest.approx(expected["mape"])
        assert report["metrics"][name]["smape"] == pytest.approx(expected["smape"])
        assert report["metrics"][name]["mase"] == pytest.approx(expected["mase"])


def test_diagnostic_prefers_seasonal_naive_on_pure_seasonality():
    pattern = np.array([1.0, 4.0, 2.0, 5.0])
    train = _frame(np.tile(pattern, 6))
    holdout = pattern.copy()
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=4
    )
    assert report["metrics"]["seasonal_naive"]["mae"] == pytest.approx(0.0)
    assert report["preferred"] in {"seasonal_naive", "ensemble"}
    assert report["metrics"]["seasonal_naive"]["mae"] <= report["metrics"]["drift"]["mae"]


def test_diagnostic_prefers_drift_on_linear_trend():
    train = _frame(np.arange(0.0, 20.0))
    holdout = np.array([20.0, 21.0, 22.0, 23.0])
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=4
    )
    assert report["metrics"]["drift"]["mae"] == pytest.approx(0.0)
    assert report["preferred"] in {"drift", "ensemble"}
    assert report["metrics"]["drift"]["mae"] <= report["metrics"]["seasonal_naive"]["mae"]


def test_diagnostic_skill_matches_direct_call():
    frame, period = _seasonal_trend_series()
    train = frame.iloc[:-4]
    holdout = frame.iloc[-4:]["value"].to_numpy()
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=period
    )
    assert report["skill"]["ensemble_vs_seasonal_naive"] == pytest.approx(
        forecast_skill_score(
            holdout,
            report["forecasts"]["ensemble"],
            report["forecasts"]["seasonal_naive"],
        )
    )


def test_diagnostic_handles_zero_mase_scale():
    train = _frame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    holdout = np.array([1.0, 1.0])
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=2
    )
    assert report["metrics"]["ensemble"]["mae"] == pytest.approx(0.0)
    assert report["metrics"]["ensemble"]["mase"] is None


def test_diagnostic_tie_prefers_ensemble():
    train = _frame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    holdout = np.array([1.0, 1.0])
    report = seasonal_naive_drift_diagnostic(
        train, "value", holdout, seasonal_period=2
    )
    assert report["metrics"]["seasonal_naive"]["mae"] == pytest.approx(0.0)
    assert report["metrics"]["drift"]["mae"] == pytest.approx(0.0)
    assert report["metrics"]["ensemble"]["mae"] == pytest.approx(0.0)
    assert report["preferred"] == "ensemble"


def test_validation_errors():
    frame, period = _seasonal_trend_series()
    holdout = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="steps must be at least 1"):
        drift_forecast(frame, "value", steps=0)
    with pytest.raises(KeyError, match="unknown target"):
        drift_forecast(frame, "missing", steps=1)
    with pytest.raises(ValueError, match="at least two observations"):
        drift_forecast(_frame([3.0]), "value", steps=1)
    with pytest.raises(ValueError, match="finite"):
        drift_forecast(_frame([1.0, np.inf, 3.0]), "value", steps=1)

    with pytest.raises(ValueError, match="seasonal_period"):
        seasonal_naive_drift_forecast(frame, "value", steps=2, seasonal_period=1)
    with pytest.raises(ValueError, match="weights"):
        seasonal_naive_drift_forecast(
            frame, "value", steps=2, seasonal_period=period, weights="median"
        )

    with pytest.raises(ValueError, match="holdout"):
        seasonal_naive_drift_diagnostic(frame, "value", [], seasonal_period=period)
    with pytest.raises(ValueError, match="match steps"):
        seasonal_naive_drift_diagnostic(
            frame, "value", holdout, steps=2, seasonal_period=period
        )
    with pytest.raises(KeyError, match="unknown target"):
        seasonal_naive_drift_diagnostic(
            frame, "missing", holdout, seasonal_period=period
        )


def test_cli_diagnose_prints_preferred(tmp_path, capsys):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = np.tile([1.0, 3.0, 2.0, 4.0], 10)
    path = tmp_path / "series.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)

    args = build_parser().parse_args(
        [str(path), "--target", "value", "--diagnose", "--seasonal-period", "4", "--steps", "4"]
    )
    report = run_cli(args)
    captured = capsys.readouterr().out
    assert "preferred:" in captured
    assert report["preferred"] in {"seasonal_naive", "drift", "ensemble"}
    assert "MAE=" in captured


def test_cli_ensemble_model_returns_kit_metrics(tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = np.arange(40, dtype=float)
    path = tmp_path / "series.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)

    args = build_parser().parse_args(
        [
            str(path),
            "--target",
            "value",
            "--model",
            "seasonal_naive_drift",
            "--seasonal-period",
            "4",
            "--steps",
            "4",
        ]
    )
    result = run_cli(args)
    assert set(result["metrics"]) >= {"mae", "rmse", "mape"}
    assert result["forecast"].shape == (4,)


def test_pipeline_includes_ensemble_and_diagnostic(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pipeline import run_pipeline

    summary = run_pipeline(output_dir=tmp_path, n_points=80)
    assert "seasonal_naive_drift" in summary["results"]
    assert {"rmse", "mae", "mape"} <= set(summary["results"]["seasonal_naive_drift"])
    diagnostic = summary["seasonal_naive_drift_diagnostic"]
    assert diagnostic["preferred"] in {"seasonal_naive", "drift", "ensemble"}
    payload = json.loads((tmp_path / "results.json").read_text())
    assert "seasonal_naive_drift_diagnostic" in payload
