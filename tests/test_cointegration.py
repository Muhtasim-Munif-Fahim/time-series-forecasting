"""Tests for Engle-Granger cointegration and the error-correction model."""

from __future__ import annotations

import math

import numpy as np
import pytest
from statsmodels.tsa.stattools import coint

from ts_forecast.cointegration import ecm_forecast, engle_granger, error_correction_model


def _pair(seed=0, n=300, rho=0.6, beta=2.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).cumsum()
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + rng.standard_normal()
    return 1.0 + beta * x + u, x


@pytest.mark.parametrize("trend", ["c", "ct", "n"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_statsmodels_coint(trend, seed):
    y, x = _pair(seed)
    if seed == 1:
        y = np.random.default_rng(99).standard_normal(300).cumsum()
    r = engle_granger(y, x, trend=trend)
    stat, p, crit = coint(y, x, trend=trend)
    assert r["statistic"] == pytest.approx(stat, rel=1e-8)
    assert r["p_value"] == pytest.approx(p, abs=1e-10)
    np.testing.assert_allclose(r["critical_values"]["5%"], crit[1])


def test_detects_cointegration_and_recovers_beta():
    y, x = _pair(3, n=500)
    r = engle_granger(y, x)
    assert r["cointegrated"]
    assert r["long_run"][0] == pytest.approx(2.0, abs=0.05)
    assert r["residuals"].shape == (500,)


def test_independent_random_walks_rarely_cointegrated():
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(1000 + s)
        y = rng.standard_normal(200).cumsum()
        x = rng.standard_normal(200).cumsum()
        hits += engle_granger(y, x)["cointegrated"]
    assert hits / 60 <= 0.15


def test_multivariate_x_and_fixed_lag():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((300, 2)).cumsum(axis=0)
    y = X @ np.array([1.0, -0.5]) + rng.standard_normal(300)
    r = engle_granger(y, X, maxlag=2, autolag=None)
    assert r["lags"] == 2
    assert r["cointegrated"]
    np.testing.assert_allclose(r["long_run"], [1.0, -0.5], atol=0.05)
    sm = coint(y, X, maxlag=2, autolag=None)
    assert r["statistic"] == pytest.approx(sm[0], rel=1e-8)


def test_ecm_speed_of_adjustment_and_half_life():
    y, x = _pair(5, n=2000, rho=0.5, beta=1.5)
    m = error_correction_model(y, x, lags=1)
    # u_t = 0.5 u_{t-1} + e  =>  du_t = -0.5 u_{t-1} + e  =>  alpha ~ -0.5
    assert m["speed_of_adjustment"] == pytest.approx(-0.5, abs=0.06)
    assert m["alpha_t_value"] < -10
    assert m["half_life"] == pytest.approx(math.log(0.5) / math.log(1 + m["speed_of_adjustment"]))
    assert m["params"]["dx0_l0"] == pytest.approx(1.5, abs=0.1)
    assert set(m["params"]) == {"const", "ec", "dx0_l0", "dx0_l1", "dy_l1"}


def test_ecm_lags_zero_and_non_cointegrated_half_life():
    rng = np.random.default_rng(6)
    y = rng.standard_normal(300).cumsum()
    x = rng.standard_normal(300).cumsum()
    m = error_correction_model(y, x, lags=0)
    assert set(m["params"]) == {"const", "ec", "dx0_l0"}
    assert abs(m["speed_of_adjustment"]) < 0.1


def test_ecm_forecast_reverts_to_equilibrium():
    y, x = _pair(7, n=400, rho=0.5, beta=2.0)
    m = error_correction_model(y, x, lags=1)
    shocked = y.copy()
    shocked[-1] += 10.0  # push y far above equilibrium
    m2 = error_correction_model(shocked, x, lags=1)
    path = ecm_forecast(m2, np.full(30, x[-1]))
    target = m2["deterministic"][0] + m2["long_run"][0] * x[-1]
    assert path.shape == (30,)
    assert abs(path[-1] - target) < abs(shocked[-1] - target) / 5
    assert np.all(np.isfinite(ecm_forecast(m, x[-5:])))


def test_validation():
    y, x = _pair(0, n=100)
    with pytest.raises(ValueError):
        engle_granger(y, x[:50])
    with pytest.raises(ValueError):
        engle_granger(y, x, trend="ctt")
    with pytest.raises(ValueError):
        engle_granger(y, x, autolag="hqic")
    with pytest.raises(ValueError):
        engle_granger(y[:10], x[:10])
    with pytest.raises(ValueError):
        engle_granger(y, np.column_stack([x, x]))
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        engle_granger(bad, x)
    with pytest.raises(ValueError):
        error_correction_model(y, x, lags=-1)
    m = error_correction_model(y, x)
    with pytest.raises(ValueError):
        ecm_forecast(m, np.ones((3, 2)))
