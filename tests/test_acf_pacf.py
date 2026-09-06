"""Tests for the ACF/PACF diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ts_forecast.evaluation import acf_pacf, plot_acf_pacf


class TestAcfPacf:
    def _series(self, *values):
        return np.array(values, dtype=float)

    def test_returns_aligned_arrays_and_metadata(self):
        values = np.random.default_rng(0).normal(size=200)
        result = acf_pacf(values, n_lags=10, alpha=0.05)

        assert set(result) == {
            "acf", "acf_conf_int", "pacf", "pacf_conf_int", "n", "alpha",
        }
        assert result["acf"].shape == (11,)
        assert result["pacf"].shape == (11,)
        assert result["acf_conf_int"].shape == (11, 2)
        assert result["pacf_conf_int"].shape == (11, 2)
        assert result["n"] == 200
        assert result["alpha"] == pytest.approx(0.05)

    def test_lag_zero_is_unit_autocorrelation(self):
        values = np.random.default_rng(1).normal(size=500)
        result = acf_pacf(values, n_lags=8)

        assert result["acf"][0] == pytest.approx(1.0)
        assert result["pacf"][0] == pytest.approx(1.0)
        assert result["acf_conf_int"][0].tolist() == [1.0, 1.0]

    def test_white_noise_stays_within_bands(self):
        values = np.random.default_rng(2).normal(size=500)
        result = acf_pacf(values, n_lags=15, alpha=0.05)

        for lag in range(1, 16):
            lower, upper = result["acf_conf_int"][lag]
            assert lower <= result["acf"][lag] <= upper

    def test_autoregressive_series_has_decaying_acf_and_single_pacf_spike(self):
        phi = 0.8
        eps = np.random.default_rng(3).normal(size=2000)
        ar = np.empty(2000)
        ar[0] = eps[0]
        for t in range(1, 2000):
            ar[t] = phi * ar[t - 1] + eps[t]

        result = acf_pacf(ar, n_lags=5, alpha=0.05)

        assert result["acf"][1] == pytest.approx(phi, abs=0.05)
        assert abs(result["pacf"][1] - phi) < 0.05
        for lag in range(2, 6):
            lower, upper = result["pacf_conf_int"][lag]
            assert lower <= result["pacf"][lag] <= upper

    def test_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            acf_pacf(self._series(1.0, np.nan, 3.0))

    def test_rejects_constant_series(self):
        with pytest.raises(ValueError, match="constant"):
            acf_pacf(self._series(5.0, 5.0, 5.0, 5.0))

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError, match="two observations"):
            acf_pacf(self._series(1.0))

    def test_rejects_negative_or_zero_n_lags(self):
        with pytest.raises(ValueError, match="positive integer"):
            acf_pacf(self._series(1.0, 2.0, 3.0), n_lags=0)

    def test_rejects_non_integer_n_lags(self):
        with pytest.raises(ValueError, match="positive integer"):
            acf_pacf(self._series(1.0, 2.0, 3.0), n_lags=2.5)

    def test_rejects_n_lags_at_least_n(self):
        with pytest.raises(ValueError, match="smaller than the number"):
            acf_pacf(self._series(1.0, 2.0, 3.0), n_lags=3)

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError, match="strictly between"):
            acf_pacf(self._series(1.0, 2.0, 3.0, 4.0), alpha=1.0)

    def test_plot_returns_two_axes_and_renders(self):
        values = np.random.default_rng(4).normal(size=200)
        fig, axes = plot_acf_pacf(values, n_lags=10, figsize=(8, 5))

        try:
            assert len(axes) == 2
            assert isinstance(fig, plt.Figure)
            for ax in axes:
                assert ax.get_xlabel() or ax.get_ylabel()
        finally:
            plt.close(fig)
