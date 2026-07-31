"""Tests for analysis module."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import format_results_table, best_model, residual_stats


class TestAnalysis:
    def test_format_table(self):
        results = {"lr": {"rmse": 1.5, "mae": 1.2, "mape": 5.0}}
        table = format_results_table(results)
        assert "lr" in table
        assert "1.5" in table

    def test_best_model(self):
        results = {"a": {"rmse": 2.0}, "b": {"rmse": 1.0}}
        assert best_model(results) == "b"

    def test_residual_stats(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        stats = residual_stats(y_true, y_pred)
        assert stats["mean"] == 0.0
        assert stats["max"] == 0.0
