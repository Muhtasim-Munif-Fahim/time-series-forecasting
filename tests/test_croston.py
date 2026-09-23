"""Tests for Croston intermittent-demand forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.models import croston_forecast, fit_croston


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _smooth(values, alpha):
    level = float(values[0])
    for observed in values[1:]:
        level += alpha * (float(observed) - level)
    return level


def test_separate_smoothing_matches_hand_calculation():
    # Demands at 1-based periods 1, 4, and 9.
    frame = _frame([5, 0, 0, 10, 0, 0, 0, 0, 20])
    fitted = fit_croston(
        frame, "value", alpha_size=0.5, alpha_interval=0.2
    )
    sizes = [5.0, 10.0, 20.0]
    intervals = [1.0, 3.0, 5.0]
    demand_size = _smooth(sizes, 0.5)
    interval = _smooth(intervals, 0.2)
    assert fitted["demand_sizes"].tolist() == sizes
    assert fitted["intervals"].tolist() == intervals
    assert fitted["demand_size"] == pytest.approx(demand_size)
    assert fitted["interval"] == pytest.approx(interval)
    assert fitted["demand_size"] == pytest.approx(13.75)
    assert fitted["interval"] == pytest.approx(2.12)
    assert fitted["rate"] == pytest.approx(13.75 / 2.12)
    assert fitted["alpha_size"] == pytest.approx(0.5)
    assert fitted["alpha_interval"] == pytest.approx(0.2)
    assert fitted["n_demands"] == 3

    forecast = croston_forecast(
        frame, "value", steps=4, alpha_size=0.5, alpha_interval=0.2
    )
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (4,)
    assert forecast.tolist() == pytest.approx([fitted["forecast_level"]] * 4)


def test_size_and_interval_update_independently():
    base = [0, 5, 0, 0, 5, 0, 0, 5]
    larger_orders = [0, 5, 0, 0, 5, 0, 0, 9]
    wider_gaps = [0, 5, 0, 0, 0, 5, 0, 0, 0, 5]
    base_fit = fit_croston(_frame(base), "value", alpha=0.3)
    larger_fit = fit_croston(_frame(larger_orders), "value", alpha=0.3)
    wider_fit = fit_croston(_frame(wider_gaps), "value", alpha=0.3)

    assert larger_fit["interval"] == pytest.approx(base_fit["interval"])
    assert larger_fit["demand_size"] != pytest.approx(base_fit["demand_size"])
    assert wider_fit["demand_size"] == pytest.approx(base_fit["demand_size"])
    assert wider_fit["interval"] != pytest.approx(base_fit["interval"])


def test_regular_sparse_pattern_forecasts_mean_rate():
    frame = _frame([0.0, 4.0] * 6)
    for alpha in (0.1, 0.4, 0.9):
        forecast = croston_forecast(frame, "value", steps=5, alpha=alpha)
        assert forecast.tolist() == pytest.approx([2.0] * 5)


def test_trailing_zeros_do_not_revise_the_interval():
    observed = [0, 0, 8, 0, 4]
    with_tail = observed + [0, 0, 0]
    left = fit_croston(_frame(observed), "value", alpha=0.25)
    right = fit_croston(_frame(with_tail), "value", alpha=0.25)
    assert right["demand_size"] == pytest.approx(left["demand_size"])
    assert right["interval"] == pytest.approx(left["interval"])
    assert right["rate"] == pytest.approx(left["rate"])


def test_single_demand_uses_its_size_and_gap():
    frame = _frame([0, 0, 8, 0])
    fitted = fit_croston(frame, "value", alpha=0.3)
    assert fitted["n_demands"] == 1
    assert fitted["demand_size"] == pytest.approx(8.0)
    assert fitted["interval"] == pytest.approx(3.0)
    forecast = croston_forecast(frame, "value", steps=2, alpha=0.3)
    assert forecast.tolist() == pytest.approx([8.0 / 3.0, 8.0 / 3.0])


def test_dense_series_reduces_to_simple_exponential_smoothing():
    values = [1.0, 3.0, 5.0, 7.0]
    fitted = fit_croston(_frame(values), "value", alpha=0.5)
    assert fitted["intervals"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert fitted["interval"] == pytest.approx(1.0)
    assert fitted["demand_size"] == pytest.approx(_smooth(values, 0.5))
    forecast = croston_forecast(_frame(values), "value", steps=3, alpha=0.5)
    assert forecast.tolist() == pytest.approx([fitted["demand_size"]] * 3)


def test_sba_scales_the_rate_by_the_interval_constant():
    frame = _frame([0.0, 4.0] * 6)
    alpha = 0.2
    classic = fit_croston(frame, "value", alpha=alpha, method="croston")
    corrected = fit_croston(frame, "value", alpha=alpha, method="sba")
    assert classic["rate"] == pytest.approx(2.0)
    assert corrected["rate"] == pytest.approx(classic["rate"])
    assert corrected["forecast_level"] == pytest.approx(
        (1.0 - alpha / 2.0) * classic["rate"]
    )
    forecast = croston_forecast(frame, "value", steps=3, alpha=alpha, method="sba")
    assert forecast.tolist() == pytest.approx([1.8, 1.8, 1.8])


def test_optimized_alphas_can_differ_by_component():
    # Constant gaps of 2, but order size jumps from 1 to 10 and stays there.
    frame = _frame([0, 1, 0, 10, 0, 10, 0, 10])
    fitted = fit_croston(frame, "value", alpha=None)
    assert fitted["alpha_interval"] == pytest.approx(0.01)
    assert fitted["alpha_size"] == pytest.approx(0.99)
    assert fitted["demand_size"] == pytest.approx(
        _smooth([1.0, 10.0, 10.0, 10.0], fitted["alpha_size"])
    )
    assert fitted["interval"] == pytest.approx(2.0)


def test_shared_alpha_is_overridden_per_component():
    frame = _frame([5, 0, 0, 10, 0, 0, 0, 0, 20])
    fitted = fit_croston(
        frame, "value", alpha=0.1, alpha_size=0.5, alpha_interval=None
    )
    assert fitted["alpha_size"] == pytest.approx(0.5)
    assert fitted["alpha_interval"] == pytest.approx(0.1)


def test_forecast_level_matches_fit():
    frame = _frame([0, 2, 0, 0, 6, 0, 3, 0, 0, 9])
    fitted = fit_croston(frame, "value", alpha_size=0.4, alpha_interval=0.15)
    forecast = croston_forecast(
        frame, "value", steps=6, alpha_size=0.4, alpha_interval=0.15
    )
    assert forecast.tolist() == pytest.approx([fitted["forecast_level"]] * 6)


def test_trailing_nan_is_dropped():
    frame = pd.DataFrame({"value": [0.0, 4.0, 0.0, 4.0, np.nan]})
    forecast = croston_forecast(frame, "value", steps=2, alpha=0.2)
    expected = croston_forecast(_frame([0.0, 4.0, 0.0, 4.0]), "value", steps=2, alpha=0.2)
    assert forecast.tolist() == pytest.approx(expected.tolist())


def test_invalid_inputs_raise():
    frame = _frame([0, 4, 0, 8, 0, 4])
    with pytest.raises(ValueError, match="steps"):
        croston_forecast(frame, "value", steps=0)
    with pytest.raises(ValueError, match="steps"):
        croston_forecast(frame, "value", steps=True)
    with pytest.raises(KeyError, match="unknown target"):
        croston_forecast(frame, "missing", steps=1)
    with pytest.raises(ValueError, match="non-negative"):
        croston_forecast(_frame([1.0, -2.0, 0.0, 3.0]), "value", steps=1)
    with pytest.raises(ValueError, match="positive demand"):
        croston_forecast(_frame([0, 0, 0, 0]), "value", steps=1)
    with pytest.raises(ValueError, match="finite"):
        croston_forecast(_frame([1.0, np.inf, 0.0]), "value", steps=1)
    with pytest.raises(ValueError, match="at least one observation"):
        croston_forecast(pd.DataFrame({"value": [np.nan, np.nan]}), "value", steps=1)
    with pytest.raises(ValueError, match="alpha"):
        croston_forecast(frame, "value", steps=1, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        croston_forecast(frame, "value", steps=1, alpha=1)
    with pytest.raises(ValueError, match="alpha_size"):
        croston_forecast(frame, "value", steps=1, alpha_size=True)
    with pytest.raises(ValueError, match="method"):
        croston_forecast(frame, "value", steps=1, method="holt")
    with pytest.raises(ValueError, match="sba"):
        fit_croston(_frame([0, 0, 5]), "value", alpha=None, method="sba")


def test_cli_croston_model_returns_kit_metrics(tmp_path):
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    values = [0.0, 4.0, 0.0, 6.0] * 5 + [2.0, 4.0, 2.0, 4.0, 2.0]
    path = tmp_path / "sparse.csv"
    pd.DataFrame({"date": dates, "value": values}).to_csv(path, index=False)

    args = build_parser().parse_args(
        [
            str(path),
            "--target",
            "value",
            "--model",
            "croston",
            "--steps",
            "4",
            "--croston-alpha",
            "0.2",
            "--croston-alpha-size",
            "0.4",
            "--croston-method",
            "sba",
        ]
    )
    result = run_cli(args)
    assert set(result["metrics"]) >= {"mae", "rmse", "mape"}
    assert result["forecast"].shape == (4,)
    assert np.all(np.isfinite(result["forecast"]))
    assert np.allclose(result["forecast"], result["forecast"][0])


def test_pipeline_includes_croston_fit(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pipeline import run_pipeline

    summary = run_pipeline(output_dir=tmp_path, n_points=80)
    assert "croston" in summary["results"]
    assert {"rmse", "mae", "mape"} <= set(summary["results"]["croston"])
    fitted = summary["croston"]
    assert fitted["method"] == "croston"
    assert fitted["n_demands"] == summary["n_train"]
    assert fitted["interval"] == pytest.approx(1.0)
    assert fitted["forecast_level"] == pytest.approx(fitted["rate"])
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["croston"]["n_demands"] == fitted["n_demands"]
    assert payload["croston"]["interval"] == pytest.approx(1.0)
