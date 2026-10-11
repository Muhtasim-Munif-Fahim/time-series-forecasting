# Time Series Forecasting Toolkit

Practical forecasting kit with statistical and ML baselines, evaluation
metrics, and a small CLI / pipeline.

Holt-Winters exponential smoothing is already available as
`holt_winters_forecast` (additive or multiplicative seasonality, optional
trend). The seasonal-naive + drift ensemble is the cheap two-component
read on whether a series is driven by **seasonality**, **trend**, or both.
The complementary Box-Jenkins check is a fixed
`SARIMA(1,1,1)(1,0,1)s` diagnostic (`fit_sarima` / `sarima_forecast`),
fit with statsmodels, which is already a dependency. Sparse demand, where
many periods are zero, is handled by Croston's method
(`fit_croston` / `croston_forecast`), which smooths order size and the
gap between orders separately.

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

## Croston intermittent demand

Seasonal-naive, drift, and SARIMA expect a demand observation in every
period. Croston's method is the counterpart for intermittent demand:
most periods are zero, and the non-zero orders arrive at irregular gaps.
`fit_croston` applies simple exponential smoothing to two series on
their own — the size of each positive demand, and the number of periods
between those demands, counting the gap from the start of the series to
the first order. `croston_forecast` repeats the per-period rate
`demand_size / interval` across the horizon.

```python
from ts_forecast.models import croston_forecast, fit_croston

fitted = fit_croston(train, "value", alpha_size=0.2, alpha_interval=0.1)
forecast = croston_forecast(train, "value", steps=8, alpha=0.1)
print(fitted["demand_size"], fitted["interval"], forecast[0])
```

`alpha` is the shared smoothing constant (default `0.1`). `alpha_size`
and `alpha_interval` override it for one component. Pass `alpha=None`
and leave an override unset to choose that component's constant by
one-step squared error. The level starts at the first demand. Zeros
only lengthen the next interval, and zeros after the last order do not
revise it. `method="sba"` applies the Syntetos-Boylan correction
`(1 - alpha_interval / 2) * demand_size / interval`.

On a series with no zeros the interval stays at 1, so the forecast
reduces to simple exponential smoothing of the observations.

## CLI

```bash
python -m ts_forecast.cli data.csv --target value --diagnose --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model seasonal_naive_drift
python -m ts_forecast.cli data.csv --target value --model holt_winters --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model sarima --seasonal-period 7
python -m ts_forecast.cli data.csv --target value --model croston --croston-alpha 0.1
python -m ts_forecast.cli data.csv --target value --model tsb --tsb-alpha 0.1
python -m ts_forecast.cli data.csv --target value --model adida --adida-base ses
python -m ts_forecast.cli data.csv --target value --model imapa --imapa-base auto
python -m ts_forecast.cli data.csv --target value --classify-demand
```

`--croston-alpha-size` and `--croston-alpha-interval` override the shared
constant. `--croston-method sba` selects the Syntetos-Boylan correction.

## Demand classification (SBC) and IMAPA

`classify_demand` places a series in one of the Syntetos-Boylan-Croston
quadrants using the average inter-demand interval (ADI) and the squared
coefficient of variation of the non-zero sizes (CV^2). The default
cut-offs are 1.32 and 0.49:

| | CV^2 <= 0.49 | CV^2 > 0.49 |
|---|---|---|
| ADI <= 1.32 | smooth (Croston) | erratic (SBA) |
| ADI > 1.32 | intermittent (SBA) | lumpy (SBA) |

`fit_imapa` / `imapa_forecast` implement IMAPA (Petropoulos & Kourentzes,
2015). IMAPA runs ADIDA at every aggregation level from `min_level` to
`max_level` and averages the per-period forecasts (or takes their median).
The default `max_level` is `max(2, ceil(ADI))`. This removes the need to
pick a single bucket size. `base_method="auto"` classifies each aggregated
series and uses Croston for smooth buckets and SBA for the rest.

```python
from ts_forecast.models import classify_demand, fit_imapa, imapa_forecast

print(classify_demand(train, "value"))   # adi, cv2, category, recommended_method
fitted = fit_imapa(train, "value", base_method="auto")
print(fitted["levels"], fitted["methods"], fitted["forecast_level"])
forecast = imapa_forecast(train, "value", steps=8)
```

## Pipeline

`python run.py` fits the existing ML / naive baselines and also scores
the seasonal-naive + drift ensemble, the Croston intermittent-demand
forecast, and, when the training series covers at least two seasons, the
SARIMA(1, 1, 1)(1, 0, 1)s diagnostic. The reports, including the smoothed
Croston demand size, interval, and rate, are written into
`output/results.json`. The synthetic pipeline series is strictly positive,
so Croston's interval smooths to 1 and the rate is simple exponential
smoothing of that series.

## TSB intermittent demand (Teunter–Syntetos–Babai)

Theta, Croston/SBA, Holt-Winters, and SARIMA already cover smooth and
intermittent baselines. TSB is the probability-based peer to Croston: every
period updates a demand **probability** (toward 1 on a hit, toward 0 on a
zero) while demand size updates only on positive observations. The flat
forecast is ``probability * demand_size``.

```python
from ts_forecast.models import fit_tsb, tsb_forecast

fitted = fit_tsb(train, "value", alpha_probability=0.2, alpha_demand=0.1)
forecast = tsb_forecast(train, "value", steps=8, alpha=0.1)
print(fitted["probability"], fitted["demand_size"], forecast[:3])
```

```bash
python -m ts_forecast.cli data.csv --target value --model tsb --tsb-alpha 0.1
```

## ADIDA temporal aggregation (Aggregate–Disaggregate Intermittent Demand)

Croston, SBA, and TSB model intermittency directly. ADIDA (Nikolopoulos et
al., 2011) removes much of it first: demand that is mostly zeros day by day is
often smooth week by week. `fit_adida` sums the series into non-overlapping
buckets of `aggregation_level` periods, aligned to the end of the training
data so the latest bucket is complete. It then forecasts the bucket totals
with a base method (`"ses"` by default, or `"croston"`, `"sba"`, `"tsb"`) and
spreads that forecast equally over the bucket's periods.

```python
from ts_forecast.models import adida_forecast, fit_adida

fitted = fit_adida(train, "value")             # bucket = rounded mean inter-demand interval
print(fitted["aggregation_level"], fitted["adi"], fitted["forecast_level"])

weekly = adida_forecast(train, "value", steps=14, aggregation_level=7, base_method="croston")
```

`aggregation_level=None` uses the rounded mean inter-demand interval (ADI),
so roughly one order lands in each bucket. Passing the season length
aggregates seasonality away instead. `aggregation_level=1` reduces to the
base method on the raw series. `alpha` is the base method's smoothing
constant: for SES it is chosen by one-step squared error when omitted, and
the Croston family defaults to `0.1`.

```bash
python -m ts_forecast.cli data.csv --target value --model adida \
  --adida-level 7 --adida-base croston --adida-alpha 0.1
```


## Fourier regression seasonality

`fourier_regression_forecast` fits a linear model with optional trend and
Fourier terms for one or more seasonal periods (Harvey / TBATS-style
seasonality without Box-Cox or ARMA). Useful when you want an explicit
harmonic seasonal pattern that extrapolates cleanly.

```python
from ts_forecast.models import fourier_regression_forecast

forecast = fourier_regression_forecast(
    train, "value", steps=14, seasonal_period=7, n_harmonics=3
)
```
