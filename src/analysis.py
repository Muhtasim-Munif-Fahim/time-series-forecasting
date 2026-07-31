"""Forecast analysis: comparison, best model selection, residual diagnostics."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def format_results_table(results: dict) -> str:
    lines = [f"{'Model':<22} {'RMSE':<10} {'MAE':<10} {'MAPE%':<10}"]
    lines.append("-" * 52)
    for model, metrics in results.items():
        lines.append(f"{model:<22} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['mape']:<10.2f}")
    return "\n".join(lines)


def load_results(path: str | Path = "output/results.json") -> dict:
    return json.loads(Path(path).read_text())


def best_model(results: dict, metric: str = "rmse") -> str:
    return min(results, key=lambda m: results[m].get(metric, float("inf")))


def residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = np.array(y_true) - np.array(y_pred)
    return {
        "mean": round(float(residuals.mean()), 4),
        "std": round(float(residuals.std()), 4),
        "min": round(float(residuals.min()), 4),
        "max": round(float(residuals.max()), 4),
        "bias": round(float(residuals.mean() / max(np.std(y_true), 1e-8)), 4),
    }
