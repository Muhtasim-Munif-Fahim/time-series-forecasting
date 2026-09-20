# Time Series Forecasting Toolkit

Practical forecasting kit with statistical and ML baselines, evaluation
metrics, and a small CLI / pipeline.

Holt-Winters exponential smoothing is already available as
`holt_winters_forecast`. When you need a cheaper read on whether a series
is driven by **seasonality**, **trend**, or both, use the seasonal-naive
+ drift ensemble diagnostic instead.

## Seasonal-naive + drift ensemble

Seasonal-naive repeats the last completed season. Drift extrapolates the
average training increment (`last + slope * h`). Their blend is a simple
ETS-like baseline that captures both pieces without fitting smoothing
parameters.

Score all three with the kit's `forecast_accuracy` battery (MAE, RMSE,
MAPE, sMAPE, bias, MASE, RMSSE):

```python
from ts_forecast.evaluation import seasonal_naive_drift_diagnostic
from ts_forecast.models import seasonal_naive_drift_forecast

forecast = seasonal_naive_drift_forecast(
    train, "value", steps=14, seasonal_period=7
)
report = seasonal_naive_drift_diagnostic(
    train, "value", test["value"].values, seasonal_period=7
)
print(report["preferred"], report["metrics"]["ensemble"]["mae"])
```

`weights="inverse_mae"` tilts the blend toward the component with the
smaller in-sample one-step error. When the training series is constant
or perfectly seasonal, MASE and RMSSE are omitted (the seasonal-naive
scale is zero) and the remaining kit metrics are still reported.

## CLI

```bash
python -m ts_forecast.cli data.csv --target value --diagnose --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model seasonal_naive_drift
python -m ts_forecast.cli data.csv --target value --model holt_winters --seasonal-period 7
```

## Pipeline

`python run.py` fits the existing ML / naive baselines and also scores
the seasonal-naive + drift ensemble, writing
`seasonal_naive_drift_diagnostic` into `output/results.json`.
