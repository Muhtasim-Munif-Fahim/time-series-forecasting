"""Tests for the Lo-MacKinlay variance ratio test and DFA."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.diagnostics import detrended_fluctuation_analysis, variance_ratio_test


def _fixed_increments():
    t = np.arange(60)
    return np.round(
        np.sin(t * 1.7) * np.cos(t * 0.31) * 3 + 0.1 * np.where(t % 7 == 0, 5, 1), 4
    )


# Reference values from arch.unitroot.VarianceRatio(debiased=True, overlap=True)
ARCH_REFERENCE = {
    2: (0.9210328753621991, -0.6116767172401764, -0.6694771143252863),
    3: (0.3492203879973882, -3.3815500575554536, -3.445313685276967),
    5: (0.1645289831172906, -2.953836107612922, -2.9659524070822645),
}


def test_variance_ratio_matches_arch_reference():
    levels = np.concatenate(([0.0], np.cumsum(_fixed_increments())))
    report = variance_ratio_test(levels, periods=(2, 3, 5))
    for row in report["periods"]:
        vr, z, z_robust = ARCH_REFERENCE[row["period"]]
        assert row["variance_ratio"] == pytest.approx(vr, rel=1e-12)
        assert row["z"] == pytest.approx(z, rel=1e-10)
        assert row["z_robust"] == pytest.approx(z_robust, rel=1e-10)
        assert row["p_value"] == pytest.approx(2 * norm.sf(abs(z)))
    assert report["n_increments"] == 60


def _brute_force(levels, q):
    x = np.diff(levels)
    T = x.size
    mu = (levels[-1] - levels[0]) / T
    sa = sum((xi - mu) ** 2 for xi in x) / (T - 1)
    m = q * (T - q + 1) * (1 - q / T)
    sc = sum((levels[t] - levels[t - q] - q * mu) ** 2 for t in range(q, T + 1)) / m
    d = [(xi - mu) ** 2 for xi in x]
    theta = 0.0
    for j in range(1, q):
        num = sum(d[t] * d[t - j] for t in range(j, T))
        theta += (2 * (q - j) / q) ** 2 * T * num / sum(d) ** 2
    return sc / sa, theta / T


def test_variance_ratio_matches_definition_and_increments_flag():
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, 1.0, size=97) * (1 + (np.arange(97) > 50))
    levels = 5.0 + np.concatenate(([0.0], np.cumsum(x)))
    a = variance_ratio_test(levels, periods=(2, 6))
    b = variance_ratio_test(x, periods=(2, 6), increments=True)
    for ra, rb, q in zip(a["periods"], b["periods"], (2, 6)):
        vr, theta = _brute_force(levels, q)
        assert ra["variance_ratio"] == pytest.approx(vr, rel=1e-12)
        assert ra["robust_variance"] == pytest.approx(theta, rel=1e-12)
        assert rb["variance_ratio"] == pytest.approx(ra["variance_ratio"], rel=1e-12)
        assert rb["z_robust"] == pytest.approx(ra["z_robust"], rel=1e-12)
    assert a["drift"] == pytest.approx(x.mean())


def test_variance_ratio_size_under_random_walk_is_close_to_nominal():
    rng = np.random.default_rng(11)
    rejections = 0
    single = 0
    trials = 300
    for _ in range(trials):
        levels = np.cumsum(rng.normal(size=600))
        report = variance_ratio_test(levels)
        rejections += report["reject_random_walk"]
        single += report["periods"][0]["p_value_robust"] < 0.05
    # Chow-Denning with Sidak is conservative; the single test is ~5%
    assert rejections / trials <= 0.08
    assert 0.02 <= single / trials <= 0.09


def test_variance_ratio_robust_to_volatility_clustering():
    rng = np.random.default_rng(5)
    homo = 0
    robust = 0
    for _ in range(200):
        n = 800
        sigma = np.where((np.arange(n) // 100) % 2 == 0, 0.3, 4.0)
        e = rng.standard_t(4, size=n) * sigma
        report = variance_ratio_test(np.cumsum(e), periods=(2,))
        row = report["periods"][0]
        homo += row["p_value"] < 0.05
        robust += row["p_value_robust"] < 0.05
    assert robust / 200 <= 0.09
    assert robust <= homo


def test_variance_ratio_detects_momentum_and_mean_reversion():
    rng = np.random.default_rng(2)
    e = rng.normal(size=3000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.3 * ar[t - 1] + e[t]
    momentum = variance_ratio_test(ar, periods=(2, 4), increments=True)
    assert momentum["periods"][0]["variance_ratio"] == pytest.approx(1.3, abs=0.06)
    assert momentum["periods"][1]["variance_ratio"] > momentum["periods"][0]["variance_ratio"]
    assert momentum["reject_random_walk"]

    stationary = rng.normal(size=3000)  # levels mean-revert instantly
    reversion = variance_ratio_test(stationary, periods=(2, 4, 8))
    assert reversion["periods"][0]["variance_ratio"] == pytest.approx(0.5, abs=0.05)
    assert all(r["z_robust"] < -5 for r in reversion["periods"])
    assert reversion["reject_random_walk"]
    worst = max(reversion["periods"], key=lambda r: abs(r["z_robust"]))
    assert reversion["chow_denning"]["period"] == worst["period"]
    assert reversion["chow_denning"]["statistic"] == pytest.approx(abs(worst["z_robust"]))
    assert reversion["chow_denning"]["p_value"] >= worst["p_value_robust"]


def test_variance_ratio_validation():
    levels = np.cumsum(np.random.default_rng(0).normal(size=50))
    with pytest.raises(ValueError, match="at least 2"):
        variance_ratio_test(levels, periods=(1,))
    with pytest.raises(ValueError, match="distinct"):
        variance_ratio_test(levels, periods=(2, 2))
    with pytest.raises(ValueError, match="2 \\* max"):
        variance_ratio_test(levels, periods=(30,))
    with pytest.raises(ValueError, match="finite"):
        variance_ratio_test(np.r_[levels, np.nan])
    with pytest.raises(ValueError, match="no variance"):
        variance_ratio_test(np.arange(50.0))
    with pytest.raises(ValueError, match="empty"):
        variance_ratio_test(levels, periods=())
    assert variance_ratio_test(levels, periods=4)["periods"][0]["period"] == 4


def _brute_dfa(x, s, order):
    y = np.cumsum(x - x.mean())
    n = y.size
    k = n // s
    res = []
    for start in [i * s for i in range(k)] + [n - (i + 1) * s for i in range(k)]:
        seg = y[start : start + s]
        t = np.arange(s)
        coef = np.polyfit(t, seg, order)
        res.extend(seg - np.polyval(coef, t))
    return np.sqrt(np.mean(np.square(res)))


def test_dfa_fluctuation_matches_definition():
    x = np.random.default_rng(4).normal(size=203)
    for order in (1, 2):
        report = detrended_fluctuation_analysis(x, scales=[5, 9, 17, 40], order=order)
        for s, f in zip(report["scales"], report["fluctuations"]):
            assert f == pytest.approx(_brute_dfa(x, int(s), order), rel=1e-9)
        slope = np.polyfit(np.log(report["scales"]), np.log(report["fluctuations"]), 1)[0]
        assert report["alpha"] == pytest.approx(slope)


def test_dfa_exponents_for_white_noise_random_walk_and_antipersistence():
    rng = np.random.default_rng(8)
    white = [detrended_fluctuation_analysis(rng.normal(size=4096))["alpha"] for _ in range(10)]
    assert np.mean(white) == pytest.approx(0.5, abs=0.04)
    walk = detrended_fluctuation_analysis(np.cumsum(rng.normal(size=4096)))
    assert walk["alpha"] == pytest.approx(1.5, abs=0.1)
    assert walk["regime"].startswith("non-stationary")
    anti = detrended_fluctuation_analysis(np.diff(rng.normal(size=4097)))
    assert anti["alpha"] < 0.25
    assert anti["regime"] == "anti-persistent"
    assert white and detrended_fluctuation_analysis(rng.normal(size=4096))["r_squared"] > 0.9


def test_dfa_detects_long_memory():
    # fractional noise via ARFIMA(0, d, 0) MA weights: Hurst = d + 0.5
    rng = np.random.default_rng(9)
    d = 0.3
    k = np.arange(1, 2000)
    psi = np.concatenate(([1.0], np.cumprod((k - 1 + d) / k)))
    e = rng.normal(size=8192 + psi.size)
    x = np.convolve(e, psi, mode="valid")[:8192]
    report = detrended_fluctuation_analysis(x, min_scale=16)
    assert report["alpha"] == pytest.approx(0.8, abs=0.08)
    assert report["regime"] == "persistent"


def test_dfa2_removes_linear_trend_that_inflates_dfa1():
    rng = np.random.default_rng(10)
    t = np.arange(2048, dtype=float)
    x = rng.normal(size=t.size) + 0.01 * t
    dfa1 = detrended_fluctuation_analysis(x, order=1)
    dfa2 = detrended_fluctuation_analysis(x, order=2)
    assert dfa1["alpha"] > 0.75
    assert dfa2["alpha"] == pytest.approx(0.5, abs=0.08)


def test_dfa_validation():
    x = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="order"):
        detrended_fluctuation_analysis(x, order=0)
    with pytest.raises(ValueError, match="exceed order"):
        detrended_fluctuation_analysis(x, scales=[2, 10], order=1)
    with pytest.raises(ValueError, match="half"):
        detrended_fluctuation_analysis(x, scales=[10, 60])
    with pytest.raises(ValueError, match="two distinct"):
        detrended_fluctuation_analysis(x, scales=[10, 10])
    with pytest.raises(ValueError, match="too short"):
        detrended_fluctuation_analysis(x[:12])
    with pytest.raises(ValueError, match="finite"):
        detrended_fluctuation_analysis(np.r_[x, np.inf])
    with pytest.raises(ValueError, match="polynomial"):
        detrended_fluctuation_analysis(np.arange(100.0), order=2)
    with pytest.raises(ValueError, match="polynomial"):
        detrended_fluctuation_analysis(np.full(100, 3.0))


def test_cli_random_walk_test(tmp_path, capsys):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.cumsum(rng.normal(size=400))})
    path = tmp_path / "walk.csv"
    df.to_csv(path, index=False)
    args = build_parser().parse_args(
        [str(path), "--target", "value", "--random-walk-test", "--vr-periods", "2", "5"]
    )
    report = run_cli(args)
    out = capsys.readouterr().out
    assert "variance ratio" in out
    assert "q=5" in out and "Chow-Denning" in out and "DFA-1" in out
    assert [r["period"] for r in report["variance_ratio"]["periods"]] == [2, 5]
    assert report["dfa"]["alpha"] > 1.2
