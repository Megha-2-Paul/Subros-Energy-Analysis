# time_series_utils.py

import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

def generate_forecast(df):
    try:
        ts = df.set_index('DATE')['TOTAL_UNIT_(UPPCL+DG)'].resample('D').sum().fillna(method='ffill')
        model = ARIMA(ts, order=(1, 1, 1)).fit()
        forecast = model.forecast(steps=7)
        fig = px.line(x=forecast.index, y=forecast.values, labels={"x": "Date", "y": "kWh"}, title="7-Day Forecast")
        return fig, forecast
    except Exception:
        return None, None

def decompose_and_plot(df):
    if 'DATE' not in df.columns or 'TOTAL_UNIT_(UPPCL+DG)' not in df.columns:
        return None

    ts = df.set_index('DATE')['TOTAL_UNIT_(UPPCL+DG)'].sort_index().resample('D').sum()
    ts = ts.interpolate(method='linear')  # Fill gaps

    try:
        decomposition = seasonal_decompose(ts, model='additive', period=7)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts.index, y=ts, name='Original',
            line=dict(color='black')))

        # Only plot if not all NaN
        if decomposition.trend.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=ts.index, y=decomposition.trend, name='Trend'))

        if decomposition.seasonal.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=ts.index, y=decomposition.seasonal, name='Seasonal'))

        if decomposition.resid.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=ts.index, y=decomposition.resid, name='Residual'))

        fig.update_layout(
            title="📈 Time Series Decomposition (Additive)",
            xaxis_title="Date",
            yaxis_title="Units",
            height=600
        )
        return fig
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None
