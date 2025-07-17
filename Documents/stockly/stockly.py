import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


image_path = "stockly.png"

st.set_page_config(page_title="Stockly", page_icon=image_path)

st.title("STOCKLY 📈")
st.subheader("Stock Price Forecast using ARIMA")

# Input
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
ticker = st.selectbox("Select Stock", tickers)
df = yf.download(ticker, start="2020-01-01", end=date.today())
data = df['Close']

model = ARIMA(data, order=(5,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='B')


d = st.date_input("Highlight a Day (Until 30 Days from Today)",
                      value=None,
                      min_value=forecast_dates[0].date(),
                      max_value=forecast_dates[-1].date())

if st.button("Forecast the Next 30 Days"):
    fig, ax = plt.subplots()
    # Show last available historical price
    ax.plot(data.index[-1:], data.iloc[-1:], label="Last Actual Price", marker='o', color='blue')
    # Show all 30 forecasted days
    ax.plot(forecast_dates, forecast, label="Forecast (Next 30 Days)", color='orange', marker='o')
    ax.set_title(f"{ticker} Forecast (Next 30 Days)")
    all_dates = list(data.index[-1:]) + list(forecast_dates)
    ax.set_xticks(all_dates)
    ax.set_xticklabels([dt.strftime('%b %d') for dt in all_dates], rotation=45, fontsize=6)

    # Annotate the selected forecast date if chosen
    if d is not None and d in [dt.date() for dt in forecast_dates]:
        idx = [dt.date() for dt in forecast_dates].index(d)
        value = forecast.iloc[idx]  # Use .iloc for integer indexing
        ax.annotate(
            f"{value:.2f}",
            (forecast_dates[idx], value),
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            color="red"
        )

    ax.legend()
    tab1, tab2 = st.tabs(["Graph", "DataFrame"])
    tab1.pyplot(fig)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': forecast})
    tab2.dataframe(forecast_df, hide_index=True, height=500)

    
    # st.pyplot(fig)


if d is not None and d in [dt.date() for dt in forecast_dates]:
    idx = [dt.date() for dt in forecast_dates].index(d)
    forecasted_price = forecast.iloc[idx].item() if hasattr(forecast.iloc[idx], 'item') else float(forecast.iloc[idx])
    current_price = data.iloc[-1].item() if hasattr(data.iloc[-1], 'item') else float(data.iloc[-1])
    pct_change = (forecasted_price - current_price) / current_price
    label = 1 if pct_change > 0.05 else 0 

    # Show in Streamlit
    st.markdown(f"""
        **Current Price**: ${current_price:.2f}
        ##### Investment Outlook on {d.strftime('%B %d, %Y')}
        **Forecasted Price**: ${forecasted_price:.2f}     
        **Expected Return**: {pct_change * 100:.2f}%  
        **Classification**: {"🟢 Good Investment" if label == 1 else "🔴 Not Recommended"}
        """)
    
    st.markdown(
        """ 
        *The dates are business days and the forecast is based on the ARIMA model fitted to the historical stock prices.*
        """
    )