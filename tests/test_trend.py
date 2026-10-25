"""Tests for Sen's slope, the seasonal Kendall test and Pettitt's test."""

import numpy as np
import pytest
from scipy.stats import theilslopes

from ts_forecast.diagnostics import mann_kendall_test
from ts_forecast.trend import (
    pettitt_test,
    seasonal_mann_kendall_test,
    sens_slope,
    trend_report,
)


def _seasonal_series(n=96, period=12, slope=0.05, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 * np.sin(2 * np.pi * t / period) + slope * t + rng.normal(size=n)


def test_sens_slope_matches_scipy_theilslopes_with_ties():
    rng = np.random.default_rng(0)
    values = np.round(rng.normal(size=60) + np.linspace(0.0, 1.0, 60), 1)
    result = sens_slope(values, alpha=0.05)
    ref = theilslopes(values, np.arange(values.size), 0.95)
    np.testing.assert_allclose(result["slope"], ref.slope, atol=1e-12)
    np.testing.assert_allclose(result["intercept"], ref.intercept, atol=1e-12)
    np.testing.assert_allclose(result["ci_low"], ref.low_slope, atol=1e-12)
    np.testing.assert_allclose(result["ci_high"], ref.high_slope, atol=1e-12)
    assert result["ci_low"] <= result["slope"] <= result["ci_high"]


def test_sens_slope_is_robust_to_outliers_and_recovers_line():
    t = np.arange(50.0)
    values = 2.0 + 0.3 * t
    values[[5, 17, 40]] += [100.0, -80.0, 60.0]
    result = sens_slope(values)
    assert result["slope"] == pytest.approx(0.3, abs=1e-12)
    assert result["intercept"] == pytest.approx(2.0, abs=0.5)


def test_sens_slope_with_custom_times_and_validation():
    times = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    result = sens_slope([1.0, 2.0, 3.0, 4.0, 5.0], times=times)
    assert result["slope"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        sens_slope([1.0, 2.0, 3.0], times=[0.0, 1.0])
    with pytest.raises(ValueError):
        sens_slope([1.0, 2.0, 3.0], times=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        sens_slope([1.0, np.nan, 3.0])


def _brute_seasonal(values, period):
    s_total, var_total, slopes = 0.0, 0.0, []
    for g in range(period):
        sub = values[g::period]
        n = sub.size
        for i in range(n):
            for j in range(i + 1, n):
                s_total += np.sign(sub[j] - sub[i])
                slopes.append((sub[j] - sub[i]) / (j - i))
        _, c = np.unique(sub, return_counts=True)
        var_total += (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18.0
    return s_total, var_total, float(np.median(slopes))


def test_seasonal_kendall_matches_brute_force_definition():
    values = np.round(_seasonal_series(), 1)
    result = seasonal_mann_kendall_test(values, period=12)
    s_ref, var_ref, slope_ref = _brute_seasonal(values, 12)
    assert result["s"] == pytest.approx(s_ref)
    assert result["var_s"] == pytest.approx(var_ref)
    assert result["slope_per_cycle"] == pytest.approx(slope_ref)
    assert result["slope_per_step"] == pytest.approx(slope_ref / 12)
    assert len(result["seasons"]) == 12
    assert sum(row["s"] for row in result["seasons"]) == pytest.approx(s_ref)


def test_seasonal_kendall_detects_trend_masked_by_seasonality():
    values = _seasonal_series(n=120, slope=0.04, seed=11)
    plain = mann_kendall_test(values)
    seasonal = seasonal_mann_kendall_test(values, period=12)
    assert seasonal["trend"] == "increasing"
    assert seasonal["p_value"] < plain["p_value"]
    lo, hi = seasonal["slope_ci_per_cycle"]
    assert lo <= 0.04 * 12 <= hi


def test_seasonal_kendall_size_on_pure_seasonal_noise():
    rejections = 0
    for seed in range(200):
        values = _seasonal_series(n=72, slope=0.0, seed=seed)
        rejections += seasonal_mann_kendall_test(values, period=12)["p_value"] < 0.05
    assert rejections / 200 < 0.09


def test_seasonal_kendall_handles_missing_values_and_validates():
    values = _seasonal_series(n=96)
    values[[3, 20, 50]] = np.nan
    result = seasonal_mann_kendall_test(values, period=12)
    assert sum(row["n"] for row in result["seasons"]) == 93
    with pytest.raises(ValueError):
        seasonal_mann_kendall_test(values, period=1)
    with pytest.raises(ValueError):
        seasonal_mann_kendall_test(values[:20], period=12)


def test_pettitt_rank_formula_matches_double_sum_with_ties():
    rng = np.random.default_rng(5)
    values = np.round(rng.normal(size=40), 0)
    result = pettitt_test(values)
    n = values.size
    brute = [
        sum(np.sign(values[j] - values[i]) for i in range(t + 1) for j in range(t + 1, n))
        for t in range(n - 1)
    ]
    np.testing.assert_allclose(result["u"], brute)
    assert result["statistic"] == pytest.approx(np.max(np.abs(brute)))


def test_pettitt_locates_mean_shift():
    rng = np.random.default_rng(9)
    values = np.concatenate([rng.normal(0.0, 1.0, 60), rng.normal(2.0, 1.0, 40)])
    result = pettitt_test(values)
    assert result["change_point"]
    assert abs(result["index"] - 59) <= 3
    assert result["shift"] > 1.0
    assert result["p_value"] < 1e-6


def test_pettitt_no_change_on_white_noise():
    rng = np.random.default_rng(1)
    result = pettitt_test(rng.normal(size=150))
    assert not result["change_point"]
    with pytest.raises(ValueError):
        pettitt_test(np.ones(20))


def test_trend_report_combines_tests():
    values = _seasonal_series(n=120, slope=0.04, seed=11)
    report = trend_report(values, period=12)
    assert report["trend"] == "increasing"
    assert report["seasonal"]["period"] == 12
    assert "Sen slope" in report["summary"]
    plain = trend_report(np.arange(30.0))
    assert plain["seasonal"] is None
    assert plain["sens_slope"]["slope"] == pytest.approx(1.0)


def test_cli_trend_test(tmp_path, capsys):
    import pandas as pd

    from ts_forecast.cli import build_parser, run_cli

    values = _seasonal_series(n=150, slope=0.05, seed=2)
    dates = pd.date_range("2024-01-01", periods=values.size, freq="D")
    pd.DataFrame({"date": dates, "value": values}).to_csv(tmp_path / "s.csv", index=False)
    args = build_parser().parse_args(
        [str(tmp_path / "s.csv"), "--target", "value", "--trend-test", "--trend-period", "12"]
    )
    report = run_cli(args)
    out = capsys.readouterr().out
    assert "Mann-Kendall" in out and "Seasonal Kendall (period 12)" in out and "Pettitt" in out
    assert report["seasonal"]["trend"] == "increasing"
