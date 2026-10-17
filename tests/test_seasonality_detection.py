"""Tests for the periodogram, Fisher's g test, and seasonal period detection."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.diagnostics import (
    _fisher_g_pvalue,
    detect_seasonal_periods,
    fisher_g_test,
    periodogram,
)


def test_periodogram_matches_scipy_and_excludes_zero_and_nyquist():
    rng = np.random.default_rng(0)
    for n in (101, 100):
        x = rng.normal(size=n)
        spec = periodogram(x, detrend=None)
        m = (n - 1) // 2
        assert spec["power"].size == m
        _, ref = signal.periodogram(x, detrend=False, scaling="spectrum", window="boxcar")
        # scipy's one-sided "spectrum" doubles interior bins and divides by n^2
        assert np.allclose(spec["power"], ref[1 : m + 1] * n / 2)
        assert np.allclose(spec["frequencies"], np.arange(1, m + 1) / n)
        assert np.allclose(spec["periods"] * spec["frequencies"], 1.0)


def test_periodogram_detrending_options():
    t = np.arange(64, dtype=float)
    line = 3.0 + 0.5 * t
    assert np.allclose(periodogram(line)["power"], 0.0, atol=1e-18)
    assert periodogram(line, detrend="constant")["power"][0] > 1.0
    with pytest.raises(ValueError, match="detrend"):
        periodogram(line, detrend="quadratic")
    with pytest.raises(ValueError, match="eight"):
        periodogram(np.arange(5.0))
    with pytest.raises(ValueError, match="finite"):
        periodogram(np.r_[np.arange(10.0), np.nan])


@pytest.mark.parametrize("m, x", [(10, 0.3), (30, 0.15), (30, 0.2), (50, 0.1)])
def test_fisher_pvalue_matches_monte_carlo(m, x):
    rng = np.random.default_rng(m)
    sims = rng.exponential(size=(100_000, m))
    g = sims.max(axis=1) / sims.sum(axis=1)
    empirical = np.mean(g >= x)
    se = np.sqrt(empirical * (1 - empirical) / g.size)
    assert abs(_fisher_g_pvalue(x, m) - empirical) < 5 * se + 1e-4


def test_fisher_pvalue_edge_cases_and_long_series_precision():
    assert _fisher_g_pvalue(0.01, 50) == 1.0  # g <= 1/m is always attained
    assert _fisher_g_pvalue(0.999, 5) == pytest.approx(5 * 0.001**4)
    # many cancelling terms: exact value equals the one-term bound to high accuracy
    m, x = 3000, 0.01
    assert _fisher_g_pvalue(x, m) == pytest.approx(m * (1 - x) ** (m - 1), rel=1e-6)
    p = _fisher_g_pvalue(0.003, 3000)
    assert 0.0 < p < 1.0


def test_fisher_g_is_calibrated_under_white_noise():
    rng = np.random.default_rng(11)
    p = np.array([fisher_g_test(rng.normal(size=96))["p_value"] for _ in range(1500)])
    assert abs(np.mean(p < 0.05) - 0.05) < 0.02
    assert abs(np.mean(p < 0.5) - 0.5) < 0.05


def test_fisher_g_detects_sinusoid_and_validates():
    rng = np.random.default_rng(1)
    t = np.arange(240)
    y = 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(size=t.size)
    result = fisher_g_test(y)
    assert result["significant"] and result["p_value"] < 1e-10
    assert result["period"] == pytest.approx(12.0)
    assert result["n_ordinates"] == 119
    limited = fisher_g_test(y, max_period=6)
    assert limited["n_ordinates"] < 119 and limited["period"] <= 6
    with pytest.raises(ValueError, match="alpha"):
        fisher_g_test(y, alpha=1.5)
    with pytest.raises(ValueError, match="max_period"):
        fisher_g_test(y, max_period=1)
    with pytest.raises(ValueError, match="variance"):
        fisher_g_test(np.ones(50))


def test_detect_weekly_cycle_with_trend_off_fourier_grid():
    rng = np.random.default_rng(0)
    t = np.arange(730)
    y = 10 + 0.02 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(size=t.size)
    report = detect_seasonal_periods(y)
    assert report["primary_period"] == 7
    # leakage into neighbouring bins is not reported as extra cycles
    assert [d["rounded_period"] for d in report["periods"]] == [7]


def test_detect_two_cycles_strongest_first():
    rng = np.random.default_rng(0)
    t = np.arange(730)
    y = 3 * np.sin(2 * np.pi * t / 7) + 4 * np.cos(2 * np.pi * t / 30.5) + rng.normal(size=t.size)
    report = detect_seasonal_periods(y)
    assert [d["rounded_period"] for d in report["periods"]] == [30, 7]
    assert detect_seasonal_periods(y, max_periods=1)["primary_period"] == 30
    assert detect_seasonal_periods(y, max_period=14)["primary_period"] == 7


def test_detect_white_noise_returns_none():
    rng = np.random.default_rng(5)
    report = detect_seasonal_periods(rng.normal(size=500))
    assert report["periods"] == [] and report["primary_period"] is None
    assert report["last_test"]["p_value"] >= 0.05
    with pytest.raises(ValueError, match="max_periods"):
        detect_seasonal_periods(rng.normal(size=50), max_periods=0)
    with pytest.raises(ValueError, match="leakage_bins"):
        detect_seasonal_periods(rng.normal(size=50), leakage_bins=-1)


def test_cli_detect_seasonality(tmp_path, capsys):
    rng = np.random.default_rng(2)
    t = np.arange(280)
    path = tmp_path / "weekly.csv"
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=t.size, freq="D"),
        "value": 50 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(size=t.size),
    }).to_csv(path, index=False)
    args = build_parser().parse_args([str(path), "--target", "value", "--detect-seasonality"])
    report = run_cli(args)
    out = capsys.readouterr().out
    assert report["primary_period"] == 7
    assert "Fisher's g" in out and "suggested --seasonal-period: 7" in out
