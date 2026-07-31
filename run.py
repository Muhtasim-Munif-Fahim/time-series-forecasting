#!/usr/bin/env python3
"""Run time series forecasting pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import run_pipeline


def main() -> int:
    print("=" * 55)
    print("Time Series Forecasting Pipeline")
    print("=" * 55)

    results = run_pipeline(output_dir="output", n_points=500)

    print(f"\nData: {results['n_points']} points ({results['n_train']} train / {results['n_test']} test)\n")
    print(f"{'Model':<22} {'RMSE':<10} {'MAE':<10} {'MAPE%':<10}")
    print("-" * 52)
    for model, metrics in results["results"].items():
        print(f"{model:<22} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['mape']:<10.2f}")

    print(f"\nResults saved to output/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
