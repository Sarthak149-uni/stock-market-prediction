import streamlit as st

from src.data_loader import get_stock_data
from src.predictor import predict_stock

st.title("AI Predictions")

ticker = st.text_input(
    "Ticker",
    "RELIANCE.NS"
)

data = get_stock_data(
    ticker,
    "2015-01-01",
    "2025-12-31"
)

predictions = predict_stock(data)

st.line_chart(predictions)