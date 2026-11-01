"""Tests for GARCH / GJR-GARCH volatility models."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.nonlinearity import arch_lm_test
from ts_forecast.volatility import fit_garch, garch_forecast, garch_value_at_risk, simulate_garch


@pytest.fixture(scope="module")
def garch_sample():
    return simulate_garch(4000, omega=0.05, alpha=0.08, beta=0.9, mu=0.03, seed=1)


@pytest.fixture(scope="module")
def garch_fit(garch_sample):
    return fit_garch(garch_sample["returns"])


def test_simulation_has_unconditional_variance_and_clustering():
    sim = simulate_garch(20000, omega=0.1, alpha=0.1, beta=0.8, seed=3)
    assert np.var(sim["returns"]) == pytest.approx(0.1 / (1 - 0.9), rel=0.1)
    assert arch_lm_test(sim["returns"], lags=4)["p_value"] < 1e-6
    np.testing.assert_allclose(sim["returns"], sim["volatility"] * sim["z"])


def test_recovers_garch_parameters(garch_fit):
    prm, se = garch_fit["params"], garch_fit["std_errors"]
    assert garch_fit["converged"]
    for name, true in (("omega", 0.05), ("alpha[1]", 0.08), ("beta[1]", 0.9), ("mu", 0.03)):
        assert abs(prm[name] - true) < 4 * se[name] + 1e-3, name
    assert garch_fit["persistence"] == pytest.approx(prm["alpha[1]"] + prm["beta[1]"])
    assert garch_fit["half_life"] == pytest.approx(math.log(0.5) / math.log(garch_fit["persistence"]))


def test_loglik_matches_direct_recursion(garch_sample, garch_fit):
    r = garch_sample["returns"]
    prm = garch_fit["params"]
    eps = r - prm["mu"]
    s2 = np.empty_like(r)
    prev_e2 = prev_s2 = np.mean((r - np.mean(r)) ** 2)  # backcast on the original scale
    for t in range(r.size):
        s2[t] = prm["omega"] + prm["alpha[1]"] * prev_e2 + prm["beta[1]"] * prev_s2
        prev_e2, prev_s2 = eps[t] ** 2, s2[t]
    ll = norm.logpdf(eps, scale=np.sqrt(s2)).sum()
    assert garch_fit["loglik"] == pytest.approx(ll, rel=1e-6)
    np.testing.assert_allclose(garch_fit["conditional_volatility"], np.sqrt(s2), rtol=1e-6)


def test_scale_invariance(garch_sample, garch_fit):
    scaled = fit_garch(garch_sample["returns"] * 0.01)
    assert scaled["params"]["omega"] == pytest.approx(garch_fit["params"]["omega"] * 1e-4, rel=1e-3)
    assert scaled["params"]["beta[1]"] == pytest.approx(garch_fit["params"]["beta[1]"], abs=1e-4)
    assert scaled["loglik"] == pytest.approx(garch_fit["loglik"] + 4000 * math.log(100), rel=1e-6)


def test_gjr_t_recovers_leverage_and_tail():
    sim = simulate_garch(5000, omega=0.05, alpha=0.03, beta=0.88, gamma=0.12, dist="t", nu=6, seed=2)
    fit = fit_garch(sim["returns"], asymmetric=True, dist="t")
    assert fit["params"]["gamma[1]"] > 0.06
    assert 4 < fit["params"]["nu"] < 9
    sym = fit_garch(sim["returns"], dist="t")
    assert fit["loglik"] > sym["loglik"]
    assert fit["aic"] < sym["aic"]


def test_forecast_mean_reverts(garch_fit):
    fc = garch_forecast(garch_fit, steps=500)
    uncond = garch_fit["unconditional_variance"]
    assert fc["variance"][-1] == pytest.approx(uncond, rel=1e-3)
    gaps = np.abs(fc["variance"] - uncond)
    assert np.all(np.diff(gaps) <= 1e-12)
    np.testing.assert_allclose(fc["cumulative_variance"], np.cumsum(fc["variance"]))
    # one-step forecast equals the recursion applied to the last observation
    prm = garch_fit["params"]
    e = garch_fit["_eps"][-1]
    s2 = garch_fit["_sigma2"][-1]
    assert fc["variance"][0] == pytest.approx(prm["omega"] + prm["alpha[1]"] * e * e + prm["beta[1]"] * s2)


def test_value_at_risk_normal_and_t(garch_fit):
    risk = garch_value_at_risk(garch_fit, steps=2, level=0.99)
    mu = garch_fit["params"]["mu"]
    np.testing.assert_allclose(risk["value_at_risk"], -(mu + risk["volatility"] * norm.ppf(0.01)))
    assert np.all(risk["expected_shortfall"] > risk["value_at_risk"])
    sim = simulate_garch(3000, omega=0.05, alpha=0.08, beta=0.9, dist="t", nu=5, seed=4)
    fit_t = fit_garch(sim["returns"], dist="t", mean="zero")
    rt = garch_value_at_risk(fit_t, level=0.975)
    nu = fit_t["params"]["nu"]
    # ES by numerical integration of the standardized-t lower tail
    c = math.sqrt((nu - 2) / nu)
    qs = np.linspace(1e-7, 0.025, 200001)
    es_num = -np.mean(student_t.ppf(qs, nu) * c) * rt["volatility"][0]
    assert rt["expected_shortfall"][0] == pytest.approx(es_num, rel=1e-3)
    assert "mu" not in fit_t["params"]


def test_higher_orders_and_pure_arch():
    sim = simulate_garch(3000, omega=0.1, alpha=[0.1, 0.05], beta=[0.5, 0.2], seed=5)
    fit = fit_garch(sim["returns"], p=2, q=2)
    assert set(fit["params"]) == {"mu", "omega", "alpha[1]", "alpha[2]", "beta[1]", "beta[2]"}
    assert fit["persistence"] < 1
    arch = fit_garch(simulate_garch(2000, omega=0.5, alpha=0.4, beta=[], seed=6)["returns"], p=0, q=1)
    assert arch["params"]["alpha[1]"] == pytest.approx(0.4, abs=0.12)
    assert garch_forecast(arch, 3)["variance"].shape == (3,)


def test_validation():
    r = np.random.default_rng(0).normal(size=200)
    with pytest.raises(ValueError):
        fit_garch(r[:5])
    with pytest.raises(ValueError):
        fit_garch(np.ones(100))
    with pytest.raises(ValueError):
        fit_garch(r, q=0)
    with pytest.raises(ValueError):
        fit_garch(r, dist="skewt")
    with pytest.raises(ValueError):
        fit_garch(r, mean="ar")
    with pytest.raises(ValueError, match="stationary"):
        simulate_garch(10, omega=0.1, alpha=0.5, beta=0.6)
    fit = fit_garch(r)
    with pytest.raises(ValueError):
        garch_forecast(fit, 0)
    with pytest.raises(ValueError):
        garch_value_at_risk(fit, level=0.3)


def test_cli_garch(tmp_path, capsys):
    sim = simulate_garch(1500, omega=1e-6, alpha=0.1, beta=0.85, seed=7)
    prices = 100 * np.exp(np.cumsum(sim["returns"]))
    path = tmp_path / "prices.csv"
    pd.DataFrame({"date": pd.date_range("2020-01-01", periods=1500, freq="D"), "close": prices}).to_csv(
        path, index=False
    )
    args = build_parser().parse_args([str(path), "--target", "close", "--garch", "--steps", "3"])
    result = run_cli(args)
    out = capsys.readouterr().out
    assert "GARCH(1,1) normal" in out and "VaR" in out
    assert result["risk"]["value_at_risk"].shape == (3,)
    args = build_parser().parse_args(
        [str(path), "--target", "close", "--garch", "--garch-asymmetric", "--garch-dist", "t", "--steps", "1"]
    )
    run_cli(args)
    assert "GJR-GARCH(1,1) t" in capsys.readouterr().out
