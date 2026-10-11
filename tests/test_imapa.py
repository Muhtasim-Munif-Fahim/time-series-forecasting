"""Tests for SBC demand classification and IMAPA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ts_forecast.cli import build_parser, run_cli
from ts_forecast.models import (
    classify_demand,
    fit_adida,
    fit_imapa,
    imapa_forecast,
)


def _frame(values):
    return pd.DataFrame({"value": np.asarray(values, dtype=float)})


def _intermittent(n=200, p=0.25, seed=0, lam=5):
    rng = np.random.default_rng(seed)
    demand = np.where(rng.random(n) < p, rng.poisson(lam, n) + 1, 0)
    return _frame(demand)


def test_classify_hand_calculation():
    # Demands at t=1, 4, 7, 10 -> intervals 2, 3, 3, 3 -> ADI 2.75.
    frame = _frame([0, 5, 0, 0, 4, 0, 0, 6, 0, 0, 5, 0])
    report = classify_demand(frame, "value")
    sizes = np.array([5.0, 4.0, 6.0, 5.0])
    assert report["adi"] == pytest.approx(2.75)
    assert report["cv2"] == pytest.approx((sizes.std(ddof=1) / sizes.mean()) ** 2)
    assert report["n_demands"] == 4
    assert report["zero_share"] == pytest.approx(8 / 12)
    assert report["category"] == "intermittent"
    assert report["recommended_method"] == "sba"


@pytest.mark.parametrize(
    "values, category",
    [
        ([5, 6, 5, 4, 6, 5, 5, 6], "smooth"),
        ([1, 30, 2, 50, 1, 40, 3, 60], "erratic"),
        ([0, 0, 5, 0, 0, 6, 0, 0, 5], "intermittent"),
        ([0, 0, 1, 0, 0, 60, 0, 0, 2, 0, 0, 80], "lumpy"),
    ],
)
def test_four_quadrants(values, category):
    report = classify_demand(_frame(values), "value")
    assert report["category"] == category
    assert report["recommended_method"] == ("croston" if category == "smooth" else "sba")


def test_cutoffs_are_inclusive_and_configurable():
    # ADI exactly 2 (every other period) is smooth once the cutoff is 2.
    frame = _frame([0, 3, 0, 3, 0, 3, 0, 3])
    assert classify_demand(frame, "value")["category"] == "intermittent"
    assert classify_demand(frame, "value", adi_cutoff=2.0)["category"] == "smooth"
    single = classify_demand(_frame([0, 0, 0, 7]), "value")
    assert single["cv2"] == 0.0
    with pytest.raises(ValueError):
        classify_demand(frame, "value", adi_cutoff=0.5)
    with pytest.raises(ValueError):
        classify_demand(frame, "value", cv2_cutoff=-1.0)
    with pytest.raises(ValueError):
        classify_demand(_frame([0, 0, 0]), "value")


def test_imapa_is_mean_of_adida_levels():
    frame = _intermittent(n=120, seed=2)
    fitted = fit_imapa(frame, "value", max_level=5, base_method="sba", alpha=0.2)
    assert fitted["levels"] == [1, 2, 3, 4, 5]
    expected = [
        fit_adida(frame, "value", aggregation_level=k, base_method="sba", alpha=0.2)[
            "forecast_level"
        ]
        for k in range(1, 6)
    ]
    assert np.allclose(fitted["level_forecasts"], expected)
    assert fitted["forecast_level"] == pytest.approx(np.mean(expected))
    median = fit_imapa(frame, "value", max_level=5, alpha=0.2, combine="median")
    assert median["forecast_level"] == pytest.approx(np.median(expected))


def test_imapa_single_level_reduces_to_adida():
    frame = _intermittent(n=90, seed=7)
    for base in ("ses", "croston", "sba", "tsb"):
        fitted = fit_imapa(frame, "value", min_level=3, max_level=3, base_method=base)
        adida = fit_adida(frame, "value", aggregation_level=3, base_method=base)
        assert fitted["forecast_level"] == pytest.approx(adida["forecast_level"])


def test_imapa_default_levels_follow_adi():
    frame = _frame([0, 5, 0, 0, 4, 0, 0, 6, 0, 0, 5, 0] * 3)
    fitted = fit_imapa(frame, "value")
    assert fitted["adi"] == pytest.approx(classify_demand(frame, "value")["adi"])
    assert fitted["levels"] == [1, 2, 3]
    smooth = fit_imapa(_frame([4, 5, 6, 5, 4, 5]), "value")
    assert smooth["levels"] == [1, 2]


def test_imapa_auto_picks_method_per_level():
    frame = _intermittent(n=240, p=0.3, seed=1)
    fitted = fit_imapa(frame, "value", max_level=8, base_method="auto")
    assert set(fitted["methods"]) <= {"croston", "sba"}
    assert fitted["methods"][0] == classify_demand(frame, "value")["recommended_method"]
    # Large buckets almost never hold zero demand, so they are smooth -> Croston.
    assert fitted["methods"][-1] == "croston"


def test_imapa_tracks_the_mean_rate_on_long_series():
    frame = _intermittent(n=3000, seed=11)
    rate = float(frame["value"].mean())
    for base in ("ses", "sba", "tsb", "auto"):
        level = fit_imapa(frame, "value", base_method=base)["forecast_level"]
        assert level == pytest.approx(rate, rel=0.35)


def test_imapa_reduces_error_versus_worst_single_level():
    errors_imapa, errors_worst = [], []
    for seed in range(12):
        full = _intermittent(n=180, p=0.2, seed=seed)
        train, test = full.iloc[:150], full.iloc[150:]
        actual = test["value"].to_numpy()
        fitted = fit_imapa(train, "value", max_level=6, alpha=0.1)
        errors_imapa.append(np.mean((actual - fitted["forecast_level"]) ** 2))
        errors_worst.append(
            max(np.mean((actual - f) ** 2) for f in fitted["level_forecasts"])
        )
    assert np.mean(errors_imapa) <= np.mean(errors_worst)


def test_imapa_forecast_shape_and_validation():
    frame = _intermittent(n=60, seed=4)
    forecast = imapa_forecast(frame, "value", steps=5)
    assert forecast.shape == (5,)
    assert np.allclose(forecast, forecast[0])
    with pytest.raises(ValueError):
        imapa_forecast(frame, "value", steps=0)
    with pytest.raises(ValueError):
        fit_imapa(frame, "value", base_method="arima")
    with pytest.raises(ValueError):
        fit_imapa(frame, "value", combine="max")
    with pytest.raises(ValueError):
        fit_imapa(frame, "value", min_level=4, max_level=2)
    with pytest.raises(ValueError):
        fit_imapa(frame, "value", max_level=1000)
    with pytest.raises(ValueError):
        fit_imapa(frame, "value", max_level=True)


def _csv(tmp_path):
    path = tmp_path / "demand.csv"
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=48, freq="D"),
        "value": [0, 5, 0, 0, 8, 0, 0, 0, 6, 0, 4, 0] * 4,
    })
    frame.to_csv(path, index=False)
    return path


def test_cli_imapa_runs(tmp_path):
    args = build_parser().parse_args([
        str(_csv(tmp_path)), "--target", "value", "--model", "imapa",
        "--steps", "4", "--test-size", "0.25", "--imapa-base", "auto",
        "--imapa-max-level", "4",
    ])
    result = run_cli(args)
    assert len(result["forecast"]) == 4
    assert result["forecast"][0] > 0


def test_cli_classify_demand(tmp_path, capsys):
    args = build_parser().parse_args([
        str(_csv(tmp_path)), "--target", "value", "--classify-demand",
    ])
    report = run_cli(args)
    out = capsys.readouterr().out
    assert "category:" in out
    assert report["category"] in {"intermittent", "lumpy"}
