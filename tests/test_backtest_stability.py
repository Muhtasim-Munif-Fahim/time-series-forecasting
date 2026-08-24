"""Tests for across-origin backtest stability metrics."""

import numpy as np
import pandas as pd
import pytest

from ts_forecast.evaluation import backtest_stability, summarize_backtest
from ts_forecast.models import rolling_origin_backtest


@pytest.fixture
def tidy_results():
    return pd.DataFrame(
        {
            "fold": [1, 1, 2, 2, 3, 3],
            "origin": [
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
            ],
            "timestamp": [
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
                "2024-01-06",
                "2024-01-06",
                "2024-01-07",
            ],
            "horizon": [1, 2, 1, 2, 1, 2],
            "actual": [4.0, 5.0, 5.0, 6.0, 6.0, 8.0],
            "prediction": [4.0, 4.5, 5.0, 5.5, 6.0, 7.0],
        }
    )


def test_backtest_stability_reports_per_horizon_dispersion(tidy_results):
    result = backtest_stability(tidy_results)

    assert result.index.tolist() == [1, 2]
    assert result.columns.tolist() == [
        "count",
        "score_mean",
        "score_std",
        "score_iqr",
    ]
    assert result["count"].tolist() == [3, 3]

    horizon_one = result.loc[1]
    assert horizon_one["score_mean"] == pytest.approx(0.0)
    assert horizon_one["score_std"] == pytest.approx(0.0)
    assert horizon_one["score_iqr"] == pytest.approx(0.0)

    horizon_two = result.loc[2]
    assert horizon_two["score_mean"] == pytest.approx(2.0 / 3.0)
    assert horizon_two["score_std"] == pytest.approx(np.sqrt(1.0 / 12.0))
    assert horizon_two["score_iqr"] == pytest.approx(0.25)


def test_backtest_stability_supports_squared_error_scores():
    results = pd.DataFrame(
        {
            "origin": ["a", "a", "b", "b"],
            "horizon": [1, 2, 1, 2],
            "actual": [10.0, 10.0, 12.0, 14.0],
            "prediction": [9.0, 13.0, 9.0, 16.0],
        }
    )

    result = backtest_stability(results, score="squared_error")
    assert result["score_mean"].tolist() == pytest.approx([5.0, 6.5])
    assert result["score_iqr"].tolist() == pytest.approx([4.0, 2.5])

    absolute = backtest_stability(results)
    assert absolute["score_mean"].tolist() == pytest.approx([2.0, 2.5])


def test_backtest_stability_single_origin_has_no_dispersion_estimate():
    results = pd.DataFrame(
        {
            "origin": ["a", "a"],
            "horizon": [1, 2],
            "actual": [1.0, 2.0],
            "prediction": [0.5, 2.5],
        }
    )

    result = backtest_stability(results)

    assert result["count"].tolist() == [1, 1]
    assert result["score_mean"].tolist() == pytest.approx([0.5, 0.5])
    assert result["score_std"].isna().tolist() == [True, True]
    assert result["score_iqr"].tolist() == [0.0, 0.0]


def test_backtest_stability_aligns_with_rolling_origin_output():
    frame = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]},
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )

    def repeat_last(train, target_col, steps):
        return np.repeat(train[target_col].iloc[-1], steps)

    results = rolling_origin_backtest(
        repeat_last, frame, "value", initial_window=3, horizon=2, step=2
    )
    stability = backtest_stability(results)

    assert stability.index.tolist() == summarize_backtest(results).index.tolist()
    assert results["fold"].unique().tolist() == [1, 2]
    assert stability["count"].tolist() == [2, 2]
    assert stability["score_mean"].tolist() == pytest.approx([3.5, 9.0])
    assert stability["score_std"].tolist() == pytest.approx([np.sqrt(4.5), np.sqrt(32.0)])
    assert stability["score_iqr"].tolist() == pytest.approx([1.5, 4.0])


def test_backtest_stability_validates_inputs(tidy_results):
    with pytest.raises(TypeError):
        backtest_stability({"origin": [], "horizon": []})

    with pytest.raises(ValueError, match="missing columns"):
        backtest_stability(tidy_results.drop(columns=["origin"]))

    with pytest.raises(ValueError, match="score must be"):
        backtest_stability(tidy_results, score="mae")


def test_backtest_stability_empty_results_keep_schema():
    empty = pd.DataFrame(
        columns=["fold", "origin", "timestamp", "horizon", "actual", "prediction"]
    )
    result = backtest_stability(empty)

    assert result.empty
    assert list(result.columns) == ["count", "score_mean", "score_std", "score_iqr"]
    assert result.index.name == "horizon"
