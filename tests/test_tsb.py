"""Tests for TSB (Teunter-Syntetos-Babai) intermittent-demand forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.models import fit_tsb, tsb_forecast


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def test_hand_calculation_probability_and_size():
    # Demands: 5 at t0, then zero, then 10.
    frame = _frame([5.0, 0.0, 10.0])
    fitted = fit_tsb(frame, "value", alpha_probability=0.5, alpha_demand=0.5)
    # init at first positive: z=5, p=1
    # t=1 zero: p = 0.5*0 + 0.5*1 = 0.5; z unchanged
    # t=2 demand 10: z = 0.5*10 + 0.5*5 = 7.5; p = 0.5*1 + 0.5*0.5 = 0.75
    assert fitted["demand_size"] == pytest.approx(7.5)
    assert fitted["probability"] == pytest.approx(0.75)
    assert fitted["forecast_level"] == pytest.approx(7.5 * 0.75)
    forecast = tsb_forecast(frame, "value", steps=3, alpha_probability=0.5, alpha_demand=0.5)
    assert forecast.tolist() == pytest.approx([fitted["forecast_level"]] * 3)


def test_trailing_zeros_pull_probability_down():
    base = _frame([0.0, 8.0, 0.0, 4.0])
    with_tail = _frame([0.0, 8.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    left = fit_tsb(base, "value", alpha=0.3)
    right = fit_tsb(with_tail, "value", alpha=0.3)
    assert right["demand_size"] == pytest.approx(left["demand_size"])
    assert right["probability"] < left["probability"]


def test_zeros_do_not_change_demand_size():
    frame = _frame([6.0, 0.0, 0.0, 0.0])
    fitted = fit_tsb(frame, "value", alpha=0.2)
    assert fitted["demand_size"] == pytest.approx(6.0)
    assert fitted["probability"] < 1.0


def test_overrides_beat_shared_alpha():
    frame = _frame([5.0, 0.0, 10.0, 0.0, 15.0])
    shared = fit_tsb(frame, "value", alpha=0.1)
    overridden = fit_tsb(
        frame, "value", alpha=0.1, alpha_probability=0.8, alpha_demand=0.2
    )
    assert overridden["alpha_probability"] == pytest.approx(0.8)
    assert overridden["alpha_demand"] == pytest.approx(0.2)
    assert overridden["probability"] != pytest.approx(shared["probability"])


def test_requires_positive_demand():
    with pytest.raises(ValueError, match="positive"):
        fit_tsb(_frame([0.0, 0.0, 0.0]), "value")


def test_rejects_bad_steps_and_alpha():
    frame = _frame([1.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        tsb_forecast(frame, "value", steps=0)
    with pytest.raises(ValueError):
        fit_tsb(frame, "value", alpha=0.0)
    with pytest.raises(ValueError):
        fit_tsb(frame, "value", alpha=1.5)


def test_cli_tsb_runs(tmp_path):
    path = tmp_path / "demand.csv"
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="D"),
        "value": [0, 5, 0, 0, 8, 0, 0, 0, 6, 0, 4, 0],
    })
    frame.to_csv(path, index=False)
    args = build_parser().parse_args([
        str(path), "--target", "value", "--model", "tsb",
        "--steps", "3", "--test-size", "0.25", "--tsb-alpha", "0.2",
    ])
    result = run_cli(args)
    assert "forecast" in result
    assert len(result["forecast"]) == 3
