"""Tests for the Kalman-filter local level / local linear trend models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.models import (
    _structural_kalman,
    _structural_loglik,
    fit_structural_model,
    structural_forecast,
    structural_prediction_interval,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _local_level(n=300, level_sd=0.5, obs_sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    level = 10.0 + np.cumsum(rng.normal(0.0, level_sd, n))
    return level + rng.normal(0.0, obs_sd, n)


def _local_trend(n=300, seed=1):
    rng = np.random.default_rng(seed)
    slope = 0.2 + np.cumsum(rng.normal(0.0, 0.05, n))
    level = np.cumsum(slope) + np.cumsum(rng.normal(0.0, 0.5, n))
    return level + rng.normal(0.0, 1.0, n)


def _brute_force_loglik(values, variances, trend):
    """Diffuse log-likelihood via the joint Gaussian of the differenced series."""
    h = variances[0]
    q = variances[1]
    n = values.size
    if not trend:
        # Delta y_t = eta_{t-1} + eps_t - eps_{t-1}: MA(1) covariance.
        z = np.diff(values)
        cov = np.diag(np.full(n - 1, q + 2 * h))
        idx = np.arange(n - 2)
        cov[idx, idx + 1] = cov[idx + 1, idx] = -h
    else:
        r = variances[2]
        # Delta^2 y_t = zeta_{t-2} + eta_{t-1} - eta_{t-2} + eps_t - 2 eps_{t-1} + eps_{t-2}.
        z = np.diff(values, 2)
        m = n - 2
        cov = np.zeros((m, m))
        lags = {0: r + 2 * q + 6 * h, 1: -q - 4 * h, 2: h}
        for lag, value in lags.items():
            idx = np.arange(m - lag)
            cov[idx, idx + lag] = value
            cov[idx + lag, idx] = value
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    quad = z @ np.linalg.solve(cov, z)
    return -0.5 * (z.size * np.log(2 * np.pi) + logdet + quad)


@pytest.mark.parametrize("trend", [False, True])
def test_loglik_matches_differenced_gaussian(trend):
    values = _local_trend(60) if trend else _local_level(60)
    variances = (0.9, 0.3, 0.01) if trend else (1.1, 0.4)
    expected = _brute_force_loglik(values, variances, trend)
    assert _structural_kalman(values, variances, trend)["loglik"] == pytest.approx(
        expected, rel=1e-10
    )
    assert _structural_loglik(values, variances, trend) == pytest.approx(
        expected, rel=1e-10
    )


@pytest.mark.parametrize("trend,spec", [(False, "llevel"), (True, "lltrend")])
def test_matches_statsmodels_unobserved_components(trend, spec):
    sm_structural = pytest.importorskip("statsmodels.tsa.statespace.structural")
    values = _local_trend() if trend else _local_level()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = sm_structural.UnobservedComponents(values, spec).fit(disp=False)
    fitted = fit_structural_model(_frame(values), "value", trend=trend)
    assert fitted["loglik"] >= reference.llf - 1e-4
    ours = list(fitted["variances"].values())
    np.testing.assert_allclose(ours, reference.params, rtol=0.02, atol=1e-4)

    pinned = structural_prediction_interval(
        _frame(values), "value", 6, trend=trend, variances=reference.params
    )
    sm_forecast = reference.get_forecast(6)
    np.testing.assert_allclose(pinned["forecast"], sm_forecast.predicted_mean, atol=1e-4)
    np.testing.assert_allclose(pinned["std"], sm_forecast.se_mean, rtol=1e-4)
    pinned_fit = fit_structural_model(
        _frame(values), "value", trend=trend, variances=reference.params
    )
    np.testing.assert_allclose(
        pinned_fit["smoothed_level"], reference.smoothed_state[0], atol=1e-4
    )


def test_zero_observation_noise_is_random_walk():
    values = _local_level(50)
    out = structural_prediction_interval(
        _frame(values), "value", 4, variances=(0.0, 2.0), coverage=0.95
    )
    np.testing.assert_allclose(out["forecast"], values[-1])
    np.testing.assert_allclose(out["std"], np.sqrt(2.0 * np.arange(1, 5)))
    z = norm.ppf(0.975)
    np.testing.assert_allclose(out["upper"] - out["forecast"], z * out["std"])


def test_zero_level_noise_forecasts_sample_mean():
    rng = np.random.default_rng(3)
    values = 5.0 + rng.normal(0.0, 1.0, 80)
    out = structural_prediction_interval(
        _frame(values), "value", 3, variances=(1.0, 0.0)
    )
    np.testing.assert_allclose(out["forecast"], values.mean(), atol=1e-12)
    np.testing.assert_allclose(out["std"], np.sqrt(1.0 + 1.0 / values.size))


def test_local_linear_trend_extrapolates_deterministic_line():
    t = np.arange(30, dtype=float)
    values = 2.0 + 0.5 * t
    forecast = structural_forecast(
        _frame(values), "value", steps=5, trend=True, variances=(1e-6, 0.0, 0.0)
    )
    np.testing.assert_allclose(forecast, 2.0 + 0.5 * np.arange(30, 35), atol=1e-6)


def test_mle_recovers_simulated_variances():
    values = _local_level(n=2000, level_sd=0.7, obs_sd=1.5, seed=11)
    fitted = fit_structural_model(_frame(values), "value")
    assert fitted["variances"]["observation"] == pytest.approx(2.25, rel=0.15)
    assert fitted["variances"]["level"] == pytest.approx(0.49, rel=0.2)
    assert fitted["signal_to_noise"] == pytest.approx(0.49 / 2.25, rel=0.3)
    resid = fitted["standardized_residuals"]
    assert resid.size == values.size - 1
    assert np.mean(resid) == pytest.approx(0.0, abs=0.1)
    assert np.std(resid) == pytest.approx(1.0, abs=0.05)
    assert np.isnan(fitted["fitted"][0]) and np.all(np.isfinite(fitted["fitted"][1:]))


def test_intervals_widen_and_nest():
    values = _local_trend(150)
    narrow = structural_prediction_interval(
        _frame(values), "value", 8, trend=True, coverage=0.5
    )
    wide = structural_prediction_interval(
        _frame(values), "value", 8, trend=True, coverage=0.95
    )
    assert np.all(np.diff(wide["std"]) > 0)
    assert np.all(wide["lower"] < narrow["lower"])
    assert np.all(wide["upper"] > narrow["upper"])
    np.testing.assert_allclose(wide["forecast"], narrow["forecast"])


def test_validation_errors():
    frame = _frame(_local_level(20))
    with pytest.raises(KeyError):
        fit_structural_model(frame, "missing")
    with pytest.raises(ValueError, match="at least 4"):
        fit_structural_model(_frame([1.0, 2.0, 3.0]), "value", trend=True)
    with pytest.raises(ValueError, match="entries"):
        fit_structural_model(frame, "value", variances=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        fit_structural_model(frame, "value", variances=(1.0, -1.0))
    with pytest.raises(ValueError, match="positive"):
        fit_structural_model(frame, "value", variances=(0.0, 0.0))
    with pytest.raises(ValueError, match="steps"):
        structural_forecast(frame, "value", steps=0)
    with pytest.raises(ValueError, match="coverage"):
        structural_prediction_interval(frame, "value", 3, coverage=1.0)


def _csv(tmp_path, values):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "value": values,
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("model", ["local_level", "local_linear_trend"])
def test_cli_structural_models(tmp_path, capsys, model):
    args = build_parser().parse_args(
        [
            str(_csv(tmp_path, _local_trend(120))),
            "--target",
            "value",
            "--model",
            model,
            "--steps",
            "6",
            "--interval-coverage",
            "0.8",
        ]
    )
    result = run_cli(args)
    out = capsys.readouterr().out
    assert "Kalman MLE" in out and "interval coverage" in out
    assert result["forecast"].shape == (6,)
    assert np.all(result["lower"] < result["upper"])


def test_cli_fourier_regression_prints_metrics(tmp_path, capsys):
    t = np.arange(84, dtype=float)
    args = build_parser().parse_args(
        [
            str(_csv(tmp_path, 10 + np.sin(2 * np.pi * t / 7))),
            "--target",
            "value",
            "--model",
            "fourier_regression",
            "--steps",
            "40",
        ]
    )
    result = run_cli(args)
    assert "Fourier regression forecast metrics" in capsys.readouterr().out
    # The horizon is clipped to the 17-row test split (84 - int(84 * 0.8)).
    assert len(result["forecast"]) == 17
