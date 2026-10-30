import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.filters import (
    CYCLE_FILTERS,
    bk_filter,
    cf_filter,
    cycle_summary,
    hamilton_filter,
    hp_filter,
    hp_lambda,
)


def _series(n=160, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 0.05 * t + np.cumsum(rng.normal(0, 0.3, n)) + 2 * np.sin(2 * np.pi * t / 16)


def test_hp_matches_statsmodels_and_penalised_objective():
    sm_hp = pytest.importorskip("statsmodels.tsa.filters.hp_filter")
    x = _series()
    for lamb in (0.0, 6.25, 1600.0, 129600.0):
        cycle, trend = sm_hp.hpfilter(x, lamb)
        out = hp_filter(x, lamb)
        np.testing.assert_allclose(out["cycle"], cycle, atol=1e-8)
        np.testing.assert_allclose(out["trend"] + out["cycle"], x)
    for n in (3, 4, 5):
        np.testing.assert_allclose(hp_filter(x[:n], 10)["cycle"], sm_hp.hpfilter(x[:n], 10)[0], atol=1e-10)
    # first-order condition of the HP objective
    tau = hp_filter(x, 1600)["trend"]
    D = np.diff(np.eye(x.size), 2, axis=0)
    np.testing.assert_allclose((tau - x) + 1600 * D.T @ (D @ tau), 0, atol=1e-7)


def test_hp_limits_and_lambda_rule():
    x = _series(80)
    np.testing.assert_allclose(hp_filter(x, 0)["trend"], x)
    t = np.arange(80.0)
    line = 3 + 0.5 * t
    np.testing.assert_allclose(hp_filter(line, 1e6)["cycle"], 0, atol=1e-8)  # linear trends pass through
    huge = hp_filter(x, 1e12)["trend"]
    np.testing.assert_allclose(huge, np.polyval(np.polyfit(t, x, 1), t), atol=1e-3)
    assert hp_lambda(4) == 1600 and hp_lambda(1) == 6.25 and hp_lambda(12) == 129600
    with pytest.raises(ValueError):
        hp_lambda(0)
    with pytest.raises(ValueError):
        hp_filter(x, -1)


def test_bk_matches_statsmodels_and_kills_quadratic_trend():
    sm_bk = pytest.importorskip("statsmodels.tsa.filters.bk_filter")
    x = _series()
    out = bk_filter(x, 6, 32, 12)
    np.testing.assert_allclose(out["cycle"][12:-12], sm_bk.bkfilter(x, 6, 32, 12), atol=1e-10)
    assert np.isnan(out["cycle"][:12]).all() and np.isnan(out["cycle"][-12:]).all()
    assert out["weights"].size == 13
    t = np.arange(100.0)
    lin = bk_filter(1 + 0.3 * t, K=8)["cycle"]
    np.testing.assert_allclose(lin[8:-8], 0, atol=1e-9)
    quad = bk_filter(1 + 0.3 * t + 0.01 * t**2, K=8)["cycle"][8:-8]
    np.testing.assert_allclose(quad, quad[0], atol=1e-9)  # quadratic -> constant (stationary)


def test_cf_matches_statsmodels_with_and_without_drift():
    sm_cf = pytest.importorskip("statsmodels.tsa.filters.cf_filter")
    x = _series(90, seed=3)
    for drift in (True, False):
        cycle, trend = sm_cf.cffilter(x, 6, 32, drift)
        out = cf_filter(x, 6, 32, drift=drift)
        np.testing.assert_allclose(out["cycle"], cycle, atol=1e-10)
        np.testing.assert_allclose(out["trend"], x - cycle, atol=1e-10)
        if not drift:
            np.testing.assert_allclose(out["trend"], trend, atol=1e-10)


def test_band_pass_isolates_business_cycle_frequency():
    t = np.arange(400)
    business = np.sin(2 * np.pi * t / 20)
    x = business + 0.8 * np.sin(2 * np.pi * t / 3) + 0.02 * t
    for out in (bk_filter(x, 6, 32, 20), cf_filter(x, 6, 32)):
        c = out["cycle"]
        inner = slice(40, 360)
        assert np.corrcoef(c[inner], business[inner])[0, 1] > 0.97
    hp_cycle = hp_filter(x, 1600)["cycle"]
    assert np.std(hp_cycle[40:360] - business[40:360]) > 0.4  # HP keeps the fast noise


def test_hamilton_matches_ols_and_alignment():
    x = _series(120, seed=5)
    h, p = 8, 4
    out = hamilton_filter(x, h, p)
    rows = np.arange(p - 1, x.size - h)
    X = np.column_stack([np.ones(rows.size)] + [x[rows - k] for k in range(p)])
    coef = np.linalg.lstsq(X, x[rows + h], rcond=None)[0]
    np.testing.assert_allclose(out["coefficients"], coef)
    assert np.isnan(out["trend"][: h + p - 1]).all() and np.isfinite(out["trend"][h + p - 1 :]).all()
    resid = out["cycle"][h + p - 1 :]
    assert abs(resid.mean()) < 1e-9  # OLS with intercept
    np.testing.assert_allclose(X.T @ resid, 0, atol=1e-7)
    # a deterministic linear trend is predicted exactly
    lin = hamilton_filter(2 + 0.5 * np.arange(60.0), h=4, p=2)["cycle"]
    np.testing.assert_allclose(lin[5:], 0, atol=1e-9)


def test_cycle_summary_and_validation():
    t = np.arange(64)
    s = cycle_summary(np.sin(2 * np.pi * t / 16 + 0.3))
    assert len(s["peaks"]) == 4 and len(s["troughs"]) == 4 and s["n"] == 64
    s2 = cycle_summary(np.r_[np.nan, 0.0, 1.0, 0.0, np.nan])
    assert s2["peaks"] == [2] and s2["troughs"] == []
    assert set(CYCLE_FILTERS) == {"hp", "bk", "cf", "hamilton"}
    x = _series(50)
    with pytest.raises(ValueError, match="low"):
        bk_filter(x, low=1.5)
    with pytest.raises(ValueError, match="high"):
        cf_filter(x, low=10, high=8)
    with pytest.raises(ValueError, match="observations"):
        bk_filter(x[:20], K=12)
    with pytest.raises(ValueError, match="finite"):
        hp_filter(np.r_[x, np.nan])
    with pytest.raises(ValueError):
        hamilton_filter(x, h=0)
    with pytest.raises(ValueError):
        cycle_summary([1.0, np.nan])


@pytest.mark.parametrize("name", ["hp", "bk", "cf", "hamilton"])
def test_cli_cycle_filter(tmp_path, capsys, name):
    dates = pd.date_range("2000-01-01", periods=200, freq="QS")
    df = pd.DataFrame({"date": dates, "value": _series(200, seed=2)})
    path = tmp_path / "gdp.csv"
    df.to_csv(path, index=False)
    args = build_parser().parse_args([str(path), "--target", "value", "--cycle-filter", name])
    report = run_cli(args)
    out = capsys.readouterr().out
    assert "cycle: n=" in out and "peaks:" in out
    assert report["filter"] == name and report["result"]["cycle"].size == 160
    if name == "hp":
        assert "lambda=1600" in out
