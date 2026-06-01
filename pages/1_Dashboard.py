import streamlit as st

from src.data_loader import get_stock_data
from src.charts import candlestick_chart
from src.indicators import add_indicators

st.title("Dashboard")

market = st.selectbox(
    "Market",
    ["NSE","BSE","US"]
)

symbol = st.text_input(
    "Stock",
    "RELIANCE"
)

if market == "NSE":
    ticker = symbol + ".NS"

elif market == "BSE":
    ticker = symbol + ".BO"

else:
    ticker = symbol

data = get_stock_data(
    ticker,
    "2015-01-01",
    "2025-12-31"
)

data = add_indicators(data)

st.plotly_chart(
    candlestick_chart(data),
    use_container_width=True
)

st.line_chart(data["MA50"])

st.line_chart(data["MA100"])

st.line_chart(data["RSI"])