import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stockly - Stock Forecast App")
st.title("STOCKLY 📈")
st.subheader("Stock Price Forecast using ARIMA")

# Input
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
ticker = st.selectbox("Select Stock Ticker", tickers)

if st.button("Predict Next 7 Days"):
    df = yf.download(ticker, start="2020-01-01", end=date.today())
    data = df['Close']

    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='B')

    fig, ax = plt.subplots()
    last_week = data.index[-5:]
    ax.plot(last_week, data.loc[last_week], label="Historical (Last Week)")
    ax.plot(forecast_dates, forecast, label="Forecast", color='orange')
    ax.set_title(f"{ticker} Forecast (Next 7 Days)")
    all_dates = list(last_week) + list(forecast_dates)
    ax.set_xticklabels([d.strftime('%B %d') for d in all_dates], rotation=45)

    ax.legend()
    st.pyplot(fig)
    
    st.markdown(
    """ 
    The dates are business days, and the forecast is based on the ARIMA model fitted to the historical stock prices.


    The detailed forecast for the next 30 days is shown below:
    """
    )
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': forecast})
    st.dataframe(forecast_df, hide_index=True)