"""Tests for Mincer-Zarnowitz, HLN encompassing and Pesaran-Timmermann tests."""

import numpy as np
import pytest

from ts_forecast.rationality import (
    forecast_encompassing_test,
    mincer_zarnowitz_test,
    pesaran_timmermann_test,
)


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = signal + rng.normal(scale=0.5, size=n)
    return rng, signal, y


@pytest.mark.parametrize("max_lag", [0, 3])
def test_mincer_zarnowitz_matches_statsmodels_hac(max_lag):
    sm = pytest.importorskip("statsmodels.api")
    _, signal, y = _series(seed=1)
    f = 0.3 + 0.8 * signal
    res = mincer_zarnowitz_test(y, f, max_lag=max_lag)
    model = sm.OLS(y, sm.add_constant(f))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lag, "use_correction": True})
    assert res["intercept"] == pytest.approx(fit.params[0])
    assert res["slope"] == pytest.approx(fit.params[1])
    assert res["se_intercept"] == pytest.approx(fit.bse[0], rel=1e-8)
    assert res["se_slope"] == pytest.approx(fit.bse[1], rel=1e-8)
    wald = fit.wald_test("const = 0, x1 = 1", scalar=True, use_f=False)
    assert res["wald_stat"] == pytest.approx(float(wald.statistic), rel=1e-8)
    assert res["p_value"] == pytest.approx(float(wald.pvalue), rel=1e-6)
    assert res["r_squared"] == pytest.approx(fit.rsquared)


def test_mincer_zarnowitz_accepts_rational_rejects_biased():
    _, signal, y = _series(seed=2)
    good = mincer_zarnowitz_test(y, signal)
    assert good["rational"]
    assert good["slope"] == pytest.approx(1.0, abs=0.1)
    biased = mincer_zarnowitz_test(y, 1.0 + 0.5 * signal)
    assert not biased["rational"]
    assert biased["p_value"] < 1e-6
    assert biased["f_p_value"] < 1e-6


def test_mincer_zarnowitz_validation():
    with pytest.raises(ValueError):
        mincer_zarnowitz_test([1, 2, 3], [1, 1, 1])
    with pytest.raises(ValueError):
        mincer_zarnowitz_test([1, 2], [1, 2])
    with pytest.raises(ValueError):
        mincer_zarnowitz_test([1, 2, 3, 4], [1, 2, 3])
    with pytest.raises(ValueError):
        mincer_zarnowitz_test(np.arange(5.0), np.arange(5.0), max_lag=-1)
    with pytest.raises(ValueError):
        mincer_zarnowitz_test([1, 2, np.nan, 4], [1, 2, 3, 4])


def test_encompassing_detects_complementary_information():
    rng = np.random.default_rng(3)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = x1 + x2 + rng.normal(scale=0.3, size=n)
    fa, fb = x1, x2
    res = forecast_encompassing_test(y, fa, fb)
    assert not res["a_encompasses_b"]
    assert res["combination_weight_b"] == pytest.approx(0.5, abs=0.1)
    assert res["mse_combination"] < min(res["mse_a"], res["mse_b"])


def test_encompassing_accepts_when_b_is_noise():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)
    fb = x + rng.normal(scale=2.0, size=n)  # noisier version of the same info
    res = forecast_encompassing_test(y, x, fb)
    assert res["a_encompasses_b"]
    assert res["combination_weight_b"] < 0.1


def test_encompassing_statistic_manual_and_corrections():
    y = np.array([1.0, 2.0, 0.5, 3.0, 2.5, 1.5])
    fa = np.array([0.5, 2.5, 0.0, 2.0, 3.0, 1.0])
    fb = np.array([1.2, 1.5, 1.0, 3.5, 2.0, 2.0])
    ea, eb = y - fa, y - fb
    d = ea * (ea - eb)
    expected = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / d.size)
    raw = forecast_encompassing_test(y, fa, fb, small_sample=False)
    assert raw["statistic"] == pytest.approx(expected)
    hln = forecast_encompassing_test(y, fa, fb, small_sample=True)
    n = d.size
    assert hln["statistic"] == pytest.approx(expected * np.sqrt((n - 1) / n))
    lagged = forecast_encompassing_test(y, fa, fb, max_lag=1)
    assert np.isfinite(lagged["statistic"])
    with pytest.raises(ValueError):
        forecast_encompassing_test(y, fa, fb[:-1])
    with pytest.raises(ValueError):
        forecast_encompassing_test(y, fa, fa)


def test_pesaran_timmermann_skill_and_chance():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.normal(size=n)
    skilled = y + rng.normal(scale=0.5, size=n)
    res = pesaran_timmermann_test(y, skilled)
    assert res["has_directional_skill"]
    assert res["hit_rate"] > res["expected_hit_rate"]
    chance = pesaran_timmermann_test(y, rng.normal(size=n))
    assert not chance["has_directional_skill"]


def test_pesaran_timmermann_manual_value_and_changes():
    y = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=float)
    f = np.array([1, -1, 1, -1, -1, 1, 1, -1], dtype=float)
    res = pesaran_timmermann_test(y, f)
    n = 8
    p = 6 / 8
    py = pf = 0.5
    p_star = 0.5
    var_p = p_star * (1 - p_star) / n
    var_star = 4 * py * pf * (1 - py) * (1 - pf) / n**2
    assert res["statistic"] == pytest.approx((p - p_star) / np.sqrt(var_p - var_star))
    levels = np.cumsum(y)
    flevels = np.concatenate([[0.0], levels[:-1] + f[1:]])
    assert pesaran_timmermann_test(levels, flevels, changes=True)["n"] == n - 1
    with pytest.raises(ValueError):
        pesaran_timmermann_test(np.ones(10), np.ones(10))
