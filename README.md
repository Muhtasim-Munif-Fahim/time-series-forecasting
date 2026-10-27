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
python -m ts_forecast.cli data.csv --target value --detect-seasonality
python -m ts_forecast.cli data.csv --target value --random-walk-test --vr-periods 2 4 8
```

`--croston-alpha-size` and `--croston-alpha-interval` override the shared
constant. `--croston-method sba` selects the Syntetos-Boylan correction.

## Seasonality detection (periodogram + Fisher's g test)

Before picking `--seasonal-period`, ask the data which cycles are real:

```python
from ts_forecast.diagnostics import periodogram, fisher_g_test, detect_seasonal_periods

spec = periodogram(values)                 # Fourier frequencies j/n, periods n/j, power
test = fisher_g_test(values)               # g = max ordinate / sum, exact p-value
report = detect_seasonal_periods(values)   # repeated g tests, strongest first
report["primary_period"]                   # e.g. 7, or None if nothing is significant
```

- `periodogram` returns `|DFT|^2 / n` at the interior Fourier frequencies
  (zero and Nyquist excluded), after removing a least-squares line by default
  (`detrend="linear" | "constant" | None`).
- `fisher_g_test` uses Fisher's (1929) exact null distribution
  `P(g >= x) = sum_k (-1)^(k-1) C(m, k) (1 - k x)^(m-1)`, evaluated in
  high-precision decimals so the alternating sum stays accurate for long
  series. Under Gaussian white noise the p-values are uniform. `max_period`
  drops slower ordinates before testing.
- `detect_seasonal_periods` removes each significant peak (plus
  `leakage_bins` neighbours, since off-grid cycles leak into adjacent bins)
  and tests the next one, up to `max_periods`. Peaks that round to a period
  already found are treated as leakage and not reported twice. Harmonics of a
  non-sinusoidal season can still appear as their own cycles.

## Random-walk and long-memory diagnostics (variance ratio, DFA)

Before trusting a model over the naive forecast, check whether the series
is distinguishable from a random walk at all:

```python
from ts_forecast.diagnostics import variance_ratio_test, detrended_fluctuation_analysis

vr = variance_ratio_test(levels, periods=(2, 4, 8, 16))
vr["periods"][0]["variance_ratio"]   # VR(2) ~ 1 + lag-1 autocorrelation of increments
vr["chow_denning"]["p_value"]        # joint test over all q (Sidak bound)
vr["reject_random_walk"]

dfa = detrended_fluctuation_analysis(values, order=1)
dfa["alpha"], dfa["regime"]          # ~0.5 white noise, ~1.5 random walk
```

- `variance_ratio_test` is Lo and MacKinlay's (1988) overlapping, bias-corrected
  variance ratio `VR(q) = Var(q-period increment) / (q Var(1-period increment))`.
  It reports the homoskedastic `z(q)` and the heteroskedasticity-robust `z*(q)`,
  so volatility clustering alone does not reject the random walk. The joint
  Chow-Denning test uses `max_q |z*(q)|` with a conservative Sidak p-value.
  `VR > 1` means momentum and `VR < 1` means mean reversion. Pass
  `increments=True` when the input is already differenced. The statistics
  match `arch.unitroot.VarianceRatio(debiased=True, overlap=True)`.
- `detrended_fluctuation_analysis` (Peng et al. 1994) integrates the
  demeaned series, removes a degree-`order` polynomial from each window
  (forward and backward windows, so no data are dropped), and fits
  `log F(s) ~ alpha log s`. DFA-2 is immune to linear trends in the series,
  which inflate DFA-1 and rescaled-range Hurst estimates.

## Nonlinear dependence and ARCH tests (BDS, ARCH-LM, McLeod-Li)

Ljung-Box only detects *linear* autocorrelation. Residuals can pass it and
still be predictable, for example through volatility clustering, threshold
dynamics or chaos. `ts_forecast.nonlinearity` adds the standard checks:

```python
from ts_forecast.nonlinearity import bds_test, arch_lm_test, mcleod_li_test, nonlinearity_report

bds = bds_test(resid, max_dim=3, distance=1.5, n_permutations=199)
bds["dimensions"][0]["statistic"], bds["dimensions"][0]["p_value_permutation"]
arch_lm_test(resid, lags=4)["p_value"]       # Engle (1982) LM = n R^2 ~ chi^2_q
mcleod_li_test(resid, lags=10)["p_value"]    # Ljung-Box on squared residuals
nonlinearity_report(resid)["verdict"]
```

- `bds_test` (Broock, Dechert, Scheinkman & LeBaron 1996) compares the
  correlation integral `C_m(eps)` of m-histories with `C_1(eps)^m`. These are
  equal under i.i.d. data. `W_m` is asymptotically N(0, 1), and the
  statistics match `statsmodels.tsa.stattools.bds` to ~1e-14. The asymptotic
  test over-rejects for a few hundred observations or fewer (Kanzler 1999),
  so `n_permutations` adds an exact-size shuffle p-value. `reject_iid` uses
  a Bonferroni correction across dimensions.
- `arch_lm_test` regresses `e_t^2` on `q` of its own lags. It matches
  `statsmodels.stats.diagnostic.het_arch` (LM and F forms).
- `mcleod_li_test` is the Ljung-Box Q on squared, demeaned residuals.
- On the CLI, `--nonlinearity-test` fits ARIMA(`--order`) to the training
  split and runs all three tests on its residuals. The related options are
  `--bds-max-dim` and `--bds-permutations`.

## Trend magnitude, seasonal Kendall and change points (Sen, Hirsch-Slack, Pettitt)

`mann_kendall_test` says whether a series trends. `ts_forecast.trend` adds
how much, whether the trend survives a seasonal cycle, and where a level
shift happened:

```python
from ts_forecast.trend import sens_slope, seasonal_mann_kendall_test, pettitt_test, trend_report

sens_slope(values)["slope"], sens_slope(values)["ci_low"]   # Theil-Sen + Gilbert (1987) CI
seasonal_mann_kendall_test(values, period=12)["p_value"]     # Hirsch, Slack & Smith (1982)
pettitt_test(values)["index"]                                # Pettitt (1979) change point
trend_report(values, period=12)["summary"]
```

- `sens_slope` is the median pairwise slope. Its slope, intercept and
  confidence bounds match `scipy.stats.theilslopes`, including with ties.
- `seasonal_mann_kendall_test` pools Mann-Kendall `S` and its tie-corrected
  variance across seasons and returns the seasonal Sen slope per cycle and
  per step. `NaN` gaps are dropped season by season. Its results match
  `pymannkendall.seasonal_test`.
- `pettitt_test` computes `U_t` from mid-ranks, which is exact with ties.
  It reports `K = max|U_t|`, the approximate p-value
  `2 exp(-6K^2/(n^3+n^2))`, and the median shift.
- On the CLI, `--trend-test` runs all of these on the training split.
  Use `--trend-period` to add the seasonal Kendall test.

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

## Local level / local linear trend (Kalman filter MLE)

`fit_structural_model` fits Harvey's basic structural models in state-space
form: the local level model (`trend=False`, a random-walk level observed
with noise; the model behind simple exponential smoothing) and the local
linear trend model (`trend=True`, adding a random-walk slope). The variances
are estimated by maximising the exact diffuse Gaussian likelihood from an
exact-diffuse Kalman filter, and the fit also returns AIC/BIC, one-step
innovations, standardized residuals, and the RTS-smoothed level.
`structural_prediction_interval` propagates the state covariance forward
for exact Gaussian intervals. Estimates, forecasts, standard errors, and
smoothed states agree with statsmodels' `UnobservedComponents` at the same
parameters. Pass `variances=(s2_obs, s2_level[, s2_slope])` to skip
estimation.

```python
from ts_forecast.models import fit_structural_model, structural_prediction_interval

fit = fit_structural_model(train, "value", trend=True)
print(fit["variances"], fit["signal_to_noise"], fit["aic"])
band = structural_prediction_interval(train, "value", 14, trend=True, coverage=0.9)
band["forecast"], band["lower"], band["upper"]
```

```bash
python -m ts_forecast.cli data.csv --target value --model local_linear_trend \
  --steps 14 --interval-coverage 0.9
```

## Singular Spectrum Analysis (SSA)

`ts_forecast.ssa` provides non-parametric SSA (Golyandina et al., 2001). The
series is embedded in an `L x K` Hankel trajectory matrix and decomposed by
SVD. Each eigentriple is mapped back to a series by diagonal averaging, and
the elementary series add up exactly to the original. Components that belong
together, such as the sine/cosine pair of a seasonal cycle, show up as blocks
in the **w-correlation** matrix. `ssa_forecast` uses recurrent (R-)forecasting.
The leading `rank` eigenvectors give a linear recurrence of order `L - 1`,
which continues the reconstructed signal. For finite-rank signals, such as a
linear trend plus a sine wave at rank 4, the continuation is exact.

```python
from ts_forecast.ssa import ssa_decompose, ssa_forecast

d = ssa_decompose(values, window=48, n_components=6,
                  groups={"trend": [0], "season": [1, 2]})
d["contribution"], d["w_correlation"], d["groups"]["season"]
forecast = ssa_forecast(train, "value", steps=24, window=48, rank=4)
```

```bash
python -m ts_forecast.cli data.csv --target value --model ssa \
  --steps 14 --ssa-window 56 --ssa-rank 4
```

If `--ssa-rank` is omitted, the rank is the smallest one that explains 90% of
`sum s^2`. A window that is a multiple of the season length separates
seasonal pairs best.
