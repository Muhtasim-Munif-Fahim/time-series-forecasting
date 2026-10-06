"""Tests for ADIDA (Aggregate-Disaggregate Intermittent Demand Approach)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.models import (
    adida_forecast,
    fit_adida,
    fit_croston,
    fit_tsb,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _intermittent(n=200, p=0.25, seed=0):
    rng = np.random.default_rng(seed)
    demand = np.where(rng.random(n) < p, rng.poisson(5, n) + 1, 0)
    return _frame(demand)


def test_hand_calculation_with_ses_and_fixed_alpha():
    # 7 periods, k=3 -> drop the oldest 1, buckets [0+4+0, 0+0+6] = [4, 6].
    frame = _frame([9.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0])
    fitted = fit_adida(frame, "value", aggregation_level=3, alpha=0.5)
    assert fitted["n_dropped"] == 1
    assert fitted["n_buckets"] == 2
    assert fitted["aggregated"].tolist() == [4.0, 6.0]
    # SES from level 4: 4 + 0.5 * (6 - 4) = 5 per bucket -> 5/3 per period.
    assert fitted["bucket_forecast"] == pytest.approx(5.0)
    assert fitted["forecast_level"] == pytest.approx(5.0 / 3.0)
    assert fitted["alpha"] == pytest.approx(0.5)
    forecast = adida_forecast(frame, "value", steps=4, aggregation_level=3, alpha=0.5)
    assert forecast.tolist() == pytest.approx([5.0 / 3.0] * 4)


def test_default_level_is_rounded_mean_inter_demand_interval():
    # Demands at t=1, 4, 7, 10 -> intervals 2, 3, 3, 3 -> ADI 2.75 -> k=3.
    frame = _frame([0, 5, 0, 0, 4, 0, 0, 6, 0, 0, 5, 0])
    fitted = fit_adida(frame, "value")
    assert fitted["adi"] == pytest.approx(2.75)
    assert fitted["aggregation_level"] == 3
    # Buckets aligned to the end: [0,5,0], [0,4,0], [0,6,0], [0,5,0].
    assert fitted["aggregated"].tolist() == [5.0, 4.0, 6.0, 5.0]
    assert fitted["n_dropped"] == 0


def test_aggregation_removes_most_zeros():
    frame = _intermittent()
    fitted = fit_adida(frame, "value")
    raw_zero_share = np.mean(frame["value"].to_numpy() == 0)
    bucket_zero_share = np.mean(fitted["aggregated"] == 0)
    assert fitted["aggregation_level"] > 1
    assert bucket_zero_share < raw_zero_share / 2


def test_level_one_reduces_to_the_base_method():
    frame = _intermittent(n=60, seed=3)
    croston = fit_adida(frame, "value", aggregation_level=1, base_method="croston", alpha=0.2)
    assert croston["forecast_level"] == pytest.approx(
        fit_croston(frame, "value", alpha=0.2)["forecast_level"]
    )
    tsb = fit_adida(frame, "value", aggregation_level=1, base_method="tsb", alpha=0.2)
    assert tsb["forecast_level"] == pytest.approx(fit_tsb(frame, "value", alpha=0.2)["forecast_level"])
    sba = fit_adida(frame, "value", aggregation_level=1, base_method="sba", alpha=0.2)
    assert sba["forecast_level"] == pytest.approx(
        fit_croston(frame, "value", alpha=0.2, method="sba")["forecast_level"]
    )


def test_per_period_level_tracks_the_mean_demand_rate():
    frame = _intermittent(n=2000, seed=5)
    mean_rate = float(frame["value"].mean())
    for base in ("ses", "croston", "sba", "tsb"):
        level = fit_adida(frame, "value", base_method=base)["forecast_level"]
        assert level == pytest.approx(mean_rate, rel=0.35), base


def test_constant_buckets_give_the_exact_rate():
    # One order of 6 every 3 periods: every bucket of 3 sums to 6 -> 2 per period.
    frame = _frame([0, 0, 6] * 10)
    for base in ("ses", "croston", "sba", "tsb"):
        fitted = fit_adida(frame, "value", base_method=base, alpha=0.3)
        assert fitted["aggregation_level"] == 3
        expected = 2.0 * (1 - 0.3 / 2) if base == "sba" else 2.0
        assert fitted["forecast_level"] == pytest.approx(expected), base


def test_ses_alpha_is_optimized_when_omitted():
    fitted = fit_adida(_intermittent(seed=7), "value", base_method="ses")
    assert fitted["alpha"] is not None
    assert 0.0 < fitted["alpha"] < 1.0


def test_seasonal_level_and_numpy_integer_level():
    frame = _intermittent(n=70, seed=2)
    weekly = fit_adida(frame, "value", aggregation_level=np.int64(7))
    assert weekly["aggregation_level"] == 7
    assert weekly["n_buckets"] == 10


def test_single_bucket_is_forecast_as_itself():
    frame = _frame([0, 3, 0, 5])
    fitted = fit_adida(frame, "value", aggregation_level=4)
    assert fitted["bucket_forecast"] == pytest.approx(8.0)
    assert fitted["forecast_level"] == pytest.approx(2.0)
    assert fitted["alpha"] is None


def test_validation():
    frame = _frame([1.0, 0.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="steps"):
        adida_forecast(frame, "value", steps=0)
    with pytest.raises(ValueError, match="base_method"):
        fit_adida(frame, "value", base_method="arima")
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="aggregation_level must be a positive integer"):
            fit_adida(frame, "value", aggregation_level=bad)
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_adida(frame, "value", aggregation_level=5)
    with pytest.raises(ValueError, match="alpha"):
        fit_adida(frame, "value", alpha=1.0)
    with pytest.raises(ValueError, match="positive demand"):
        fit_adida(_frame([0.0, 0.0, 0.0]), "value")
    with pytest.raises(ValueError, match="non-negative"):
        fit_adida(_frame([1.0, -1.0]), "value")
    with pytest.raises(KeyError):
        fit_adida(frame, "missing")
    # Every order sits in the dropped leading period -> no positive bucket.
    with pytest.raises(ValueError, match="positive bucket"):
        fit_adida(_frame([5.0, 0.0, 0.0]), "value", aggregation_level=2, base_method="croston")
    # SES is happy to forecast zero in the same situation.
    zero = fit_adida(_frame([5.0, 0.0, 0.0]), "value", aggregation_level=2)
    assert zero["forecast_level"] == pytest.approx(0.0)


def test_cli_adida_runs(tmp_path):
    path = tmp_path / "demand.csv"
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=24, freq="D"),
        "value": [0, 5, 0, 0, 8, 0, 0, 0, 6, 0, 4, 0] * 2,
    })
    frame.to_csv(path, index=False)
    args = build_parser().parse_args([
        str(path), "--target", "value", "--model", "adida",
        "--steps", "4", "--test-size", "0.25",
        "--adida-level", "3", "--adida-base", "croston", "--adida-alpha", "0.2",
    ])
    result = run_cli(args)
    assert len(result["forecast"]) == 4
    assert np.allclose(result["forecast"], result["forecast"][0])
    assert result["forecast"][0] > 0
