"""Visualization utilities for time series analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast(train, test, forecast, target_col, title="Forecast Results", figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train.index, train[target_col], label="Training Data", color="blue", alpha=0.6)
    if test is not None and len(test) > 0:
        ax.plot(test.index[:len(forecast)], test[target_col].values[:len(forecast)], label="Actual", color="green")
    ax.plot(pd.date_range(train.index[-1], periods=len(forecast)+1, freq="D")[1:], forecast.values, label="Forecast", color="red", linestyle="--", marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_residuals(y_true, y_pred, figsize=(12, 4)):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Residual Distribution")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals vs Predicted")
    return fig, (ax1, ax2)


def plot_decomposition(
    values,
    seasonal_period,
    model="additive",
    title=None,
    figsize=(12, 8),
):
    """Plot trend, seasonal, and residual components from :func:`seasonal_decompose`.

    The four-panel figure shows the original observed series alongside the
    extracted trend (centered moving average), seasonal pattern, and residual
    noise. NaN edges from the centered moving average are masked so the trend
    panel focuses on the well-defined interior.
    """

    from ts_forecast.evaluation import seasonal_decompose

    observed = np.asarray(values, dtype=float).ravel()
    components = seasonal_decompose(observed, seasonal_period, model=model)
    trend = components["trend"]
    seasonal = components["seasonal"]
    residual = components["residual"]

    x = np.arange(observed.size)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(x, observed, color="steelblue")
    axes[0].set_ylabel("Observed")
    axes[0].set_title(title or f"Seasonal Decomposition ({model})")

    finite_trend = np.isfinite(trend)
    axes[1].plot(x[finite_trend], trend[finite_trend], color="darkorange")
    axes[1].set_ylabel("Trend")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, seasonal, color="green")
    axes[2].set_ylabel("Seasonal")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, residual, color="gray")
    axes[3].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes
