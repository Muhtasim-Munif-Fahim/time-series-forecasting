"""Tests for the SARIMA(1, 1, 1)(1, 0, 1)s diagnostic wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.evaluation import forecast_accuracy, sarima_diagnostic
from ts_forecast.models import (
    SARIMA_DIAGNOSTIC_ORDER,
    SARIMA_DIAGNOSTIC_SEASONAL_ORDER,
    fit_sarima,
    sarima_forecast,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _seasonal_trend(n=48, period=4, slope=0.4, level=20.0):
    time = np.arange(n, dtype=float)
    season = np.array([0.0, 3.0, 1.0, 4.0])
    return level + slope * time + season[time.astype(int) % period]


def test_forecast_reproduces_a_synthetic_seasonal_pattern():
    pattern = np.array([10.0, 12.0, 11.0, 15.0])
    frame = _frame(np.tile(pattern, 8))
    forecast = sarima_forecast(frame, "value", steps=4, seasonal_period=4)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (4,)
    assert forecast.tolist() == pytest.approx(pattern.tolist(), abs=0.05)


def test_fit_exposes_fixed_order_and_aic():
    frame = _frame(np.tile([10.0, 12.0, 11.0, 15.0], 6))
    fitted = fit_sarima(frame, "value", seasonal_period=4)
    forecast = np.asarray(fitted.forecast(4), dtype=float)
    assert forecast.shape == (4,)
    assert np.isfinite(fitted.aic)
    assert SARIMA_DIAGNOSTIC_ORDER == (1, 1, 1)
    assert SARIMA_DIAGNOSTIC_SEASONAL_ORDER == (1, 0, 1)


def test_constant_series_stays_flat():
    frame = _frame(np.full(16, 5.0))
    forecast = sarima_forecast(frame, "value", steps=3, seasonal_period=4)
    assert forecast.tolist() == pytest.approx([5.0, 5.0, 5.0], abs=1e-6)


def test_trailing_nan_is_dropped():
    values = np.tile([1.0, 3.0, 2.0, 4.0], 4).tolist() + [np.nan]
    frame = pd.DataFrame({"value": values})
    forecast = sarima_forecast(frame, "value", steps=4, seasonal_period=4)
    assert forecast.shape == (4,)
    assert np.all(np.isfinite(forecast))


def test_diagnostic_prefers_sarima_on_trend_plus_season():
    values = _seasonal_trend()
    train, holdout = _frame(values[:-8]), values[-8:]
    report = sarima_diagnostic(train, "value", holdout, seasonal_period=4)
    assert report["order"] == (1, 1, 1)
    assert report["seasonal_order"] == (1, 0, 1, 4)
    assert report["forecasts"]["sarima"].shape == (8,)
    assert set(report["metrics"]) == {"sarima", "seasonal_naive", "drift", "ensemble"}
    direct = forecast_accuracy(
        holdout,
        report["forecasts"]["sarima"],
        y_train=train["value"].to_numpy(),
        seasonal_period=4,
    )
    assert report["metrics"]["sarima"]["mae"] == pytest.approx(direct["mae"])
    assert report["metrics"]["sarima"]["mae"] < report["metrics"]["seasonal_naive"]["mae"]
    assert report["skill"]["sarima_vs_seasonal_naive"] > 0
    assert report["preferred"] == "sarima"
    assert np.isfinite(report["aic"])


def test_diagnostic_tie_prefers_sarima_on_a_constant_series():
    frame = _frame(np.full(20, 4.0))
    report = sarima_diagnostic(frame, "value", np.full(4, 4.0), seasonal_period=4)
    for name in ("sarima", "seasonal_naive", "drift", "ensemble"):
        assert report["metrics"][name]["mae"] == pytest.approx(0.0, abs=1e-6)
        assert report["metrics"][name]["mase"] is None
    assert report["preferred"] == "sarima"
    assert report["skill"]["sarima_vs_seasonal_naive"] is None


def test_validates_steps_period_column_and_length():
    frame = _frame(np.tile([1.0, 2.0, 3.0, 4.0], 4))
    with pytest.raises(ValueError, match="steps must be at least 1"):
        sarima_forecast(frame, "value", steps=0, seasonal_period=4)
    with pytest.raises(ValueError, match="seasonal_period must be an integer"):
        sarima_forecast(frame, "value", steps=1, seasonal_period=True)
    with pytest.raises(ValueError, match="at least 2"):
        sarima_forecast(frame, "value", steps=1, seasonal_period=1)
    with pytest.raises(KeyError, match="unknown target column"):
        sarima_forecast(frame, "missing", steps=1, seasonal_period=4)
    with pytest.raises(ValueError, match="two complete seasonal"):
        sarima_forecast(_frame([1.0, 2.0, 3.0, 4.0]), "value", steps=1, seasonal_period=4)
    with pytest.raises(ValueError, match="at least one observation"):
        sarima_forecast(pd.DataFrame({"value": [np.nan, np.nan]}), "value", steps=1, seasonal_period=2)
    with pytest.raises(ValueError, match="finite values"):
        sarima_forecast(
            pd.DataFrame({"value": np.tile([1.0, np.inf, 3.0, 4.0], 2)}),
            "value",
            steps=1,
            seasonal_period=4,
        )


def test_diagnostic_validates_holdout():
    frame = _frame(np.tile([1.0, 2.0, 3.0, 4.0], 4))
    with pytest.raises(ValueError, match="at least one holdout"):
        sarima_diagnostic(frame, "value", [], seasonal_period=4)
    with pytest.raises(ValueError, match="y_true length must match steps"):
        sarima_diagnostic(frame, "value", [1.0, 2.0], steps=4, seasonal_period=4)
    with pytest.raises(ValueError, match="finite values"):
        sarima_diagnostic(frame, "value", [1.0, np.nan], seasonal_period=4)


def test_cli_sarima_model_returns_kit_metrics(tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    values = _seasonal_trend(n=40, slope=0.2)
    path = tmp_path / "series.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)

    args = build_parser().parse_args(
        [
            str(path),
            "--target",
            "value",
            "--model",
            "sarima",
            "--seasonal-period",
            "4",
            "--steps",
            "4",
        ]
    )
    result = run_cli(args)
    assert set(result["metrics"]) >= {"mae", "rmse", "mape"}
    assert result["forecast"].shape == (4,)
    assert np.all(np.isfinite(result["forecast"]))


def test_pipeline_includes_sarima_diagnostic(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pipeline import run_pipeline

    summary = run_pipeline(output_dir=tmp_path, n_points=80)
    assert "sarima" in summary["results"]
    assert {"rmse", "mae", "mape"} <= set(summary["results"]["sarima"])
    diagnostic = summary["sarima_diagnostic"]
    assert diagnostic["order"] == [1, 1, 1]
    assert diagnostic["seasonal_order"] == [1, 0, 1, 30]
    assert diagnostic["preferred"] in {"sarima", "seasonal_naive", "drift", "ensemble"}
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["sarima_diagnostic"]["preferred"] == diagnostic["preferred"]
