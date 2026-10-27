"""Tests for Singular Spectrum Analysis decomposition and forecasting."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.ssa import fit_ssa, ssa_decompose, ssa_forecast, w_correlation


def _signal(n):
    t = np.arange(n, dtype=float)
    return 5 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12)


def test_components_sum_to_series_and_match_svd():
    x = np.random.default_rng(0).normal(size=80).cumsum()
    d = ssa_decompose(x, window=20)
    np.testing.assert_allclose(d["components"].sum(axis=0), x, atol=1e-10)
    traj = np.array([x[i : i + 61] for i in range(20)])
    np.testing.assert_allclose(d["singular_values"], np.linalg.svd(traj, compute_uv=False))
    assert d["contribution"].sum() == pytest.approx(1.0)
    assert d["window"] == 20 and d["components"].shape == (20, 80)


def test_hankelization_matches_naive_diagonal_averaging():
    x = np.random.default_rng(1).normal(size=15)
    L = 6
    traj = np.array([x[i : i + 10] for i in range(L)])
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    elem = s[0] * np.outer(U[:, 0], Vt[0])
    naive = np.array([np.mean([elem[i, k - i] for i in range(L) if 0 <= k - i < 10]) for k in range(15)])
    d = ssa_decompose(x, window=L, n_components=1)
    np.testing.assert_allclose(d["components"][0], naive)


def test_trend_plus_sine_separates_into_groups():
    x = _signal(144)
    d = ssa_decompose(x, window=48, n_components=4, groups={"trend": [0, 3], "season": [1, 2]})
    t = np.arange(144.0)
    # rank-4 signal: the two groups add up to it exactly; trend and sine are
    # only approximately separable, so each group is close but not exact
    np.testing.assert_allclose(d["groups"]["season"] + d["groups"]["trend"], x, atol=1e-8)
    np.testing.assert_allclose(d["groups"]["season"], 3 * np.sin(2 * np.pi * t / 12), atol=0.25)
    np.testing.assert_allclose(d["groups"]["trend"], 5 + 0.1 * t, atol=0.25)
    # sine pair: similar singular values and a strong w-correlation block
    s = d["singular_values"]
    assert s[1] / s[2] < 1.05
    W = d["w_correlation"]
    assert abs(W[1, 2]) > 0.9 and abs(W[0, 1]) < 0.1
    np.testing.assert_allclose(np.diag(W), 1.0)


def test_recurrent_forecast_is_exact_for_finite_rank_signal():
    x = _signal(130)
    train = pd.DataFrame({"y": x[:100]})
    fc = ssa_forecast(train, "y", steps=30, window=48, rank=4)
    np.testing.assert_allclose(fc, x[100:], atol=1e-8)
    model = fit_ssa(train, "y", window=48, rank=4)
    assert model["lrf"].shape == (47,) and model["verticality"] < 1
    assert model["sigma"] < 1e-8


def test_noisy_forecast_beats_naive_and_auto_rank():
    rng = np.random.default_rng(2)
    x = _signal(200) + rng.normal(scale=0.5, size=200)
    train = pd.DataFrame({"y": x[:176]})
    fc = ssa_forecast(train, "y", steps=24, window=72, rank=4)
    truth = _signal(200)[176:]
    naive = np.full(24, x[175])
    assert np.mean(np.abs(fc - truth)) < 0.5 * np.mean(np.abs(naive - truth))
    auto = fit_ssa(train, "y", window=72, variance_threshold=0.99)
    assert auto["rank"] >= 3


def test_w_correlation_of_orthogonal_rows():
    t = np.arange(48.0)
    C = np.vstack([np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12), np.zeros(48)])
    W = w_correlation(C, 12)
    assert W.shape == (3, 3) and abs(W[0, 1]) < 0.05 and W[2, 0] == 0.0


def test_validation():
    x = _signal(40)
    df = pd.DataFrame({"y": x})
    with pytest.raises(ValueError):
        ssa_decompose(x[:3])
    with pytest.raises(ValueError):
        ssa_decompose(np.array([1.0, np.nan, 2.0, 3.0, 4.0]))
    for w in (1, 40, 2.5):
        with pytest.raises(ValueError):
            ssa_decompose(x, window=w)
    with pytest.raises(ValueError):
        ssa_decompose(x, n_components=0)
    with pytest.raises(ValueError):
        ssa_decompose(x, n_components=2, groups={"bad": [5]})
    with pytest.raises(ValueError):
        ssa_forecast(df, "y", steps=0)
    with pytest.raises(ValueError):
        fit_ssa(df, "y", rank=0)
    with pytest.raises(ValueError):
        fit_ssa(df, "y", variance_threshold=0.0)


def test_cli_ssa(tmp_path, capsys):
    values = _signal(120)
    path = tmp_path / "series.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=120, freq="D"), "value": values}).to_csv(path, index=False)
    args = build_parser().parse_args(
        [str(path), "--target", "value", "--model", "ssa", "--steps", "12", "--ssa-window", "36", "--ssa-rank", "4"]
    )
    result = run_cli(args)
    assert "SSA recurrent forecast metrics" in capsys.readouterr().out
    assert result["forecast"].shape == (12,)
    np.testing.assert_allclose(result["forecast"], values[96:108], atol=1e-6)
