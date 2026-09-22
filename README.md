# Time Series Forecasting Toolkit

Practical forecasting kit with statistical and ML baselines, evaluation
metrics, and a small CLI / pipeline.

Holt-Winters exponential smoothing is already available as
`holt_winters_forecast` (additive or multiplicative seasonality, optional
trend). The seasonal-naive + drift ensemble is the cheap two-component
read on whether a series is driven by **seasonality**, **trend**, or both.
The complementary Box-Jenkins check is a fixed
`SARIMA(1,1,1)(1,0,1)s` diagnostic (`fit_sarima` / `sarima_forecast`),
fit with statsmodels, which is already a dependency.

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

## SARIMA(1, 1, 1)(1, 0, 1)s diagnostic

Holt-Winters already covers triple exponential smoothing, so this kit
does not add a second ETS implementation. `fit_sarima` fits the fixed
order SARIMA(1, 1, 1)(1, 0, 1, s) and `sarima_forecast` returns the
horizon. `sarima_diagnostic` scores that forecast against seasonal-naive,
drift, and their equal-weight ensemble with the same `forecast_accuracy`
metrics used by the seasonal-naive + drift diagnostic.

```python
from ts_forecast.evaluation import sarima_diagnostic
from ts_forecast.models import fit_sarima, sarima_forecast

forecast = sarima_forecast(train, "value", steps=8, seasonal_period=4)
fitted = fit_sarima(train, "value", seasonal_period=4)
report = sarima_diagnostic(
    train, "value", test["value"].values, seasonal_period=4
)
print(report["preferred"], report["metrics"]["sarima"]["mae"], fitted.aic)
```

`report["seasonal_order"]` is `(1, 0, 1, s)`. `preferred` is the lowest-MAE
component; exact ties prefer SARIMA, then the ensemble, then
seasonal-naive, then drift. `skill` is the percent improvement of the
SARIMA forecast over each baseline.

## CLI

```bash
python -m ts_forecast.cli data.csv --target value --diagnose --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model seasonal_naive_drift
python -m ts_forecast.cli data.csv --target value --model holt_winters --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model sarima --seasonal-period 7
```

## Pipeline

`python run.py` fits the existing ML / naive baselines and also scores
the seasonal-naive + drift ensemble and, when the training series covers
at least two seasons, the SARIMA(1, 1, 1)(1, 0, 1)s diagnostic. Both
reports are written into `output/results.json`.
