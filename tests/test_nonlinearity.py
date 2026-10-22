import warnings

import numpy as np
import pandas as pd
import pytest

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.nonlinearity import arch_lm_test, bds_test, mcleod_li_test, nonlinearity_report


def _garch(n=600, seed=0):
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    h = np.ones(n)
    for t in range(1, n):
        h[t] = 0.1 + 0.3 * e[t - 1] ** 2 + 0.6 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    return e


def _logistic(n=300):
    z = np.empty(n)
    z[0] = 0.3
    for t in range(1, n):
        z[t] = 3.99 * z[t - 1] * (1 - z[t - 1])
    return z


@pytest.mark.parametrize("n,distance", [(60, 0.5), (250, 1.5), (250, 2.0)])
def test_bds_matches_statsmodels(n, distance):
    from statsmodels.tsa.stattools import bds

    x = np.random.default_rng(n).standard_normal(n)
    ours = bds_test(x, max_dim=4, distance=distance)
    stat, p = bds(x, max_dim=4, distance=distance)
    np.testing.assert_allclose([r["statistic"] for r in ours["dimensions"]], stat, atol=1e-10)
    np.testing.assert_allclose([r["p_value"] for r in ours["dimensions"]], p, atol=1e-10)
    assert [r["dimension"] for r in ours["dimensions"]] == [2, 3, 4]


def test_bds_detects_dependence_that_ljung_box_misses():
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Nonlinear MA: x_t = e_t + 0.8 e_{t-1} e_{t-2} is serially uncorrelated but dependent.
    e = np.random.default_rng(0).standard_normal(402)
    x = e[2:] + 0.8 * e[1:-1] * e[:-2]
    assert acorr_ljungbox(x, lags=[5])["lb_pvalue"].iloc[0] > 0.5
    report = bds_test(x)
    assert report["reject_iid"]
    assert report["dimensions"][1]["statistic"] > 4
    chaos = bds_test(_logistic())
    assert chaos["reject_iid"] and chaos["dimensions"][0]["statistic"] > 5


def test_bds_permutation_pvalues_and_iid_case():
    x = np.random.default_rng(3).standard_normal(120)
    report = bds_test(x, n_permutations=49, seed=1)
    assert report["p_value_used"] == "p_value_permutation"
    for row in report["dimensions"]:
        assert 1 / 50 <= row["p_value_permutation"] <= 1
    assert not report["reject_iid"]
    again = bds_test(x, n_permutations=49, seed=1)
    assert again["dimensions"] == report["dimensions"]
    chaotic = bds_test(_logistic(120), n_permutations=49)
    assert chaotic["dimensions"][0]["p_value_permutation"] == pytest.approx(1 / 50)


def test_bds_epsilon_and_validation():
    x = np.random.default_rng(4).standard_normal(100)
    sd = x.std(ddof=1)
    assert bds_test(x, epsilon=sd)["epsilon"] == pytest.approx(bds_test(x, distance=1.0)["epsilon"])
    for kwargs in ({"max_dim": 1}, {"max_dim": 40}, {"epsilon": 0.0}, {"distance": -1}, {"n_permutations": -1}, {"alpha": 1.0}):
        with pytest.raises(ValueError):
            bds_test(x, **kwargs)
    with pytest.raises(ValueError):
        bds_test(np.ones(50))
    with pytest.raises(ValueError):
        bds_test(np.arange(10.0))
    with pytest.raises(ValueError):
        bds_test(np.r_[x, np.nan])


def test_arch_lm_matches_statsmodels_het_arch():
    from statsmodels.stats.diagnostic import het_arch

    e = _garch()
    for lags in (1, 4, 8):
        ours = arch_lm_test(e, lags=lags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm, lm_p, f, f_p = het_arch(e, nlags=lags)
        assert ours["lm_statistic"] == pytest.approx(lm, rel=1e-10)
        assert ours["p_value"] == pytest.approx(lm_p, rel=1e-8, abs=1e-300)
        assert ours["f_statistic"] == pytest.approx(f, rel=1e-10)
        assert ours["f_p_value"] == pytest.approx(f_p, rel=1e-8, abs=1e-300)
        assert ours["arch_effects"] and ours["nobs"] == e.size - lags
    shifted = arch_lm_test(e + 3.0, lags=4, demean=True)
    assert shifted["lm_statistic"] == pytest.approx(arch_lm_test(e - e.mean(), lags=4)["lm_statistic"])
    with pytest.raises(ValueError):
        arch_lm_test(e, lags=0)
    with pytest.raises(ValueError):
        arch_lm_test(e[:12], lags=6)


def test_mcleod_li_matches_ljung_box_on_squares():
    from statsmodels.stats.diagnostic import acorr_ljungbox

    e = _garch(seed=2)
    ours = mcleod_li_test(e, lags=10)
    ref = acorr_ljungbox((e - e.mean()) ** 2, lags=10, return_df=True)
    np.testing.assert_allclose([r["statistic"] for r in ours["lags"]], ref["lb_stat"], rtol=1e-10)
    np.testing.assert_allclose([r["p_value"] for r in ours["lags"]], ref["lb_pvalue"], rtol=1e-8, atol=1e-300)
    assert ours["arch_effects"]
    with pytest.raises(ValueError):
        mcleod_li_test(e[:10], lags=10)


def test_arch_tests_have_reasonable_size_under_iid():
    rng = np.random.default_rng(11)
    arch_rej = ml_rej = 0
    for _ in range(300):
        x = rng.standard_normal(300)
        arch_rej += arch_lm_test(x, lags=4)["arch_effects"]
        ml_rej += mcleod_li_test(x, lags=10)["arch_effects"]
    assert arch_rej / 300 < 0.09 and ml_rej / 300 < 0.09


def test_report_verdicts():
    iid = nonlinearity_report(np.random.default_rng(5).standard_normal(300))
    assert iid["verdict"].startswith("no evidence")
    assert nonlinearity_report(_garch())["verdict"].startswith("conditional heteroskedasticity")
    tent = 1 - 2 * np.abs(_logistic() - 0.5)
    assert set(nonlinearity_report(tent)) == {"bds", "arch_lm", "mcleod_li", "verdict"}


def test_cli_nonlinearity_test(tmp_path, capsys):
    e = _garch(500, seed=7)
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    pd.DataFrame({"date": dates, "value": 50 + np.cumsum(e)}).to_csv(tmp_path / "s.csv", index=False)
    args = build_parser().parse_args(
        [str(tmp_path / "s.csv"), "--target", "value", "--order", "0,1,0",
         "--nonlinearity-test", "--bds-max-dim", "2", "--bds-permutations", "9"]
    )
    report = run_cli(args)
    out = capsys.readouterr().out
    assert "BDS m=2" in out and "ARCH-LM(4)" in out and "McLeod-Li(10)" in out and "perm p=" in out
    assert report["arch_lm"]["arch_effects"]
    assert len(report["bds"]["dimensions"]) == 1
