"""Tests for top-down hierarchical forecast reconciliation."""

import numpy as np
import pytest

from ts_forecast.models import reconcile_top_down


def test_children_receive_their_share_of_the_total():
    forecasts = {"total": [100.0], "north": [70.0], "south": [50.0]}
    structure = {"total": ["north", "south"]}
    shares = {"north": 0.6, "south": 0.4}

    result = reconcile_top_down(forecasts, structure, shares)

    assert result["north"].tolist() == [60.0]
    assert result["south"].tolist() == [40.0]
    assert result["total"].tolist() == [100.0]
    assert forecasts["north"] == [70.0]


def test_branch_forecasts_are_discarded_and_aggregates_derived():
    forecasts = {
        "total": [200.0],
        "region_a": [999.0],
        "store_1": [1.0],
        "store_2": [2.0],
    }
    structure = {
        "total": ["region_a", "region_b"],
        "region_a": ["store_1", "store_2"],
    }
    shares = {
        "region_a": 0.5,
        "region_b": 0.5,
        "store_1": 0.25,
        "store_2": 0.75,
    }

    result = reconcile_top_down(forecasts, structure, shares)

    assert set(result) == {"total", "region_a", "region_b", "store_1", "store_2"}
    assert result["store_1"].tolist() == [25.0]
    assert result["store_2"].tolist() == [75.0]
    assert result["region_a"].tolist() == [100.0]
    assert result["region_b"].tolist() == [100.0]
    assert result["total"].tolist() == [200.0]


def test_sibling_shares_are_normalized_elementwise():
    forecasts = {"total": [100.0, 300.0], "north": [0.0, 0.0], "south": [0.0, 0.0]}
    structure = {"total": ["north", "south"]}
    shares = {"north": 7.0, "south": 3.0}

    result = reconcile_top_down(forecasts, structure, shares)

    assert result["north"].tolist() == pytest.approx([70.0, 210.0])
    assert result["south"].tolist() == pytest.approx([30.0, 90.0])


def test_standalone_nodes_pass_through():
    forecasts = {"total": [10.0], "leaf": [9.0], "other": [42.0]}
    structure = {"total": ["leaf"]}
    shares = {"leaf": 1.0}

    result = reconcile_top_down(forecasts, structure, shares)

    assert result["other"].tolist() == [42.0]


def test_independent_hierarchies_reconcile_separately():
    forecasts = {
        "total_a": [80.0],
        "total_b": [20.0],
        "x": [5.0],
        "x2": [5.0],
        "y": [5.0],
    }
    structure = {"total_a": ["x", "x2"], "total_b": ["y"]}
    shares = {"x": 0.25, "x2": 0.75, "y": 1.0}

    result = reconcile_top_down(forecasts, structure, shares)

    assert result["x"].tolist() == [20.0]
    assert result["x2"].tolist() == [60.0]
    assert result["y"].tolist() == [20.0]
    assert result["total_a"].tolist() == [80.0]
    assert result["total_b"].tolist() == [20.0]


def test_derived_leaves_need_no_forecast_of_their_own():
    forecasts = {"total": [100.0]}
    structure = {"total": ["region"]}
    shares = {"region": 1.0}

    result = reconcile_top_down(forecasts, structure, shares)

    assert set(result) == {"total", "region"}
    assert result["region"].tolist() == [100.0]


def test_validates_inputs():
    with pytest.raises(ValueError, match="mapping of node names"):
        reconcile_top_down([1.0], {}, {})
    with pytest.raises(ValueError, match="at least one node"):
        reconcile_top_down({}, {"total": ["leaf"]}, {"leaf": 1.0})
    with pytest.raises(ValueError, match="aggregates to their children"):
        reconcile_top_down({"a": [1.0]}, {}, {})
    with pytest.raises(ValueError, match="proportions"):
        reconcile_top_down({"a": [1.0]}, {"a": ["b"]}, None)
    with pytest.raises(ValueError, match="same horizon"):
        reconcile_top_down(
            {"a": [1.0, 2.0], "b": [1.0]},
            {"c": ["a", "b"]},
            {"a": 1.0, "b": 1.0},
        )
    with pytest.raises(KeyError, match="missing share"):
        reconcile_top_down(
            {"a": [1.0], "b": [1.0]}, {"a": ["b"]}, {}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        reconcile_top_down(
            {"a": [1.0], "b": [1.0]}, {"a": ["b"]}, {"b": -0.5}
        )
    with pytest.raises(ValueError, match="positive share sum"):
        reconcile_top_down(
            {"a": [1.0], "b": [1.0]}, {"a": ["b"]}, {"b": 0.0}
        )
    with pytest.raises(ValueError, match="cycle"):
        reconcile_top_down(
            {"a": [1.0], "b": [1.0]}, {"a": ["b"], "b": ["a"]}, {"a": 1.0, "b": 1.0}
        )
    with pytest.raises(KeyError, match="missing forecast for top-level node"):
        reconcile_top_down(
            {"north": [1.0]}, {"total": ["north"]}, {"north": 1.0}
        )
