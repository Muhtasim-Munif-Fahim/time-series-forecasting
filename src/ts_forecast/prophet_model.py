"""Prophet-based time series forecasting."""

import pandas as pd
from prophet import Prophet


def forecast_prophet(train, target_col, steps=30, yearly_seasonality=True, weekly_seasonality=True):
    df = train.reset_index()
    df = df.rename(columns={df.columns[0]: "ds", target_col: "y"})
    model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps)
