import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# TITLE
# ======================================================

st.title("📈 AI Stock Market Prediction Platform")
st.markdown(
"""
Indian & Global Stock Analysis using
Deep Learning (LSTM), Technical Indicators,
Candlestick Charts and News Analytics.
"""
)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Market Selection")

market = st.sidebar.selectbox(
    "Choose Market",
    ["NSE", "BSE", "US", "INDEX"]
)

if market == "INDEX":

    index = st.sidebar.selectbox(
        "Choose Index",
        [
            "NIFTY50",
            "BANKNIFTY",
            "SENSEX"
        ]
    )

    if index == "NIFTY50":
        ticker = "^NSEI"

    elif index == "BANKNIFTY":
        ticker = "^NSEBANK"

    else:
        ticker = "^BSESN"

else:

    popular = {
        "Reliance": "RELIANCE",
        "TCS": "TCS",
        "Infosys": "INFY",
        "HDFC Bank": "HDFCBANK",
        "SBI": "SBIN",
        "ITC": "ITC",
        "Tata Motors": "TATAMOTORS"
    }

    company = st.sidebar.selectbox(
        "Popular Stocks",
        list(popular.keys())
    )

    symbol = st.sidebar.text_input(
        "Or Enter Symbol",
        popular[company]
    )

    if market == "NSE":
        ticker = symbol.upper() + ".NS"

    elif market == "BSE":
        ticker = symbol.upper() + ".BO"

    else:
        ticker = symbol.upper()

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

# ======================================================
# DOWNLOAD DATA
# ======================================================

with st.spinner("Downloading Stock Data..."):

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )
    # Fix yfinance MultiIndex issue

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

for col in ["Open", "High", "Low", "Close"]:

    if col in data.columns:

        if isinstance(data[col], pd.DataFrame):

            data[col] = data[col].iloc[:, 0]

if data.empty:
    st.error("No stock data found.")
    st.stop()

# ======================================================
# STOCK INFO
# ======================================================

st.subheader(f"Selected Symbol : {ticker}")

col1, col2, col3 = st.columns(3)

latest_price = float(np.array(data["Close"]).flatten()[-1])

day_high = float(np.array(data["High"]).flatten()[-1])

day_low = float(np.array(data["Low"]).flatten()[-1])

col1.metric("Current Price", f"₹ {latest_price:.2f}")

col2.metric("Day High", f"₹ {day_high:.2f}")

col3.metric("Day Low", f"₹ {day_low:.2f}")

# ======================================================
# RAW DATA
# ======================================================

with st.expander("View Raw Data"):
    st.dataframe(data.tail(20))

# ======================================================
# CANDLESTICK CHART
# ======================================================

st.subheader("Candlestick Chart")

fig = go.Figure(
    data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )
    ]
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# MOVING AVERAGES
# ======================================================

st.subheader("Moving Averages")

data["MA50"] = data["Close"].rolling(50).mean()
data["MA100"] = data["Close"].rolling(100).mean()

fig_ma = go.Figure()

fig_ma.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Close"
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA50"],
        name="MA50"
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MA100"],
        name="MA100"
    )
)

st.plotly_chart(
    fig_ma,
    use_container_width=True
)

# ======================================================
# RSI
# ======================================================

st.subheader("RSI Indicator")

delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)

loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

st.line_chart(rsi)

# ======================================================
# AI PREDICTION
# ======================================================

st.subheader("AI Prediction")

try:
    if len(data) < 120:
        st.warning(
            "Not enough historical data for prediction."
        )
        st.stop()
        
    model = load_model(
        "model/stock_lstm_model.keras"
    )

    close_data = data[["Close"]]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        close_data
    )

    x_test = []
    y_test = []

    for i in range(100, len(scaled_data)):

        x_test.append(
            scaled_data[i-100:i]
        )

        y_test.append(
            scaled_data[i]
        )

    x_test = np.array(x_test)

    y_test = np.array(y_test)

    predictions = model.predict(
        x_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions
    )

    actual_prices = scaler.inverse_transform(
        y_test
    )

    fig_pred = go.Figure()

    dates = data.index[-len(actual_prices):]

    fig_pred.add_trace(
        go.Scatter(
            x=dates,
            y=actual_prices.flatten(),
            name="Actual"
        )
    )

    fig_pred.add_trace(
        go.Scatter(
            x=dates,
            y=predictions.flatten(),
            name="Predicted"
        )
   )

    st.plotly_chart(
        fig_pred,
        use_container_width=True
    )

    last_100 = scaled_data[-100:]

    future_input = np.reshape(
        last_100,
        (1, 100, 1)
    )

    next_day = model.predict(
        future_input,
        verbose=0
    )

    next_day_price = scaler.inverse_transform(
        next_day
    )[0][0]

    st.success(
        f"Predicted Next Trading Day Price : ₹ {next_day_price:.2f}"
    )

except Exception as e:

    st.warning(
        "Train and place your model inside model/stock_lstm_model.keras"
    )

    st.error(e)

# ======================================================
# NEWS SECTION
# ======================================================

st.subheader("Latest Market News")

try:

    search_term = ticker.replace(
        ".NS",
        ""
    ).replace(
        ".BO",
        ""
    )

    feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={search_term}+stock"
    )

    for article in feed.entries[:10]:

        st.write(article.title)

        st.link_button(
            f"Read: {article.title[:40]}",
            article.link
        )

except:
    st.warning(
        "Unable to fetch news."
    )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
"""
### Tech Stack

- Python
- Streamlit
- TensorFlow
- LSTM
- Plotly
- Yahoo Finance
- Scikit-Learn

Developed for Indian & Global Stock Market Analysis.
"""
)