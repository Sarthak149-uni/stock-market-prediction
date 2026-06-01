import streamlit as st

from src.news import get_news

st.title("Market News")

stock = st.text_input(
    "Stock",
    "RELIANCE"
)

feed = get_news(stock)

for article in feed.entries[:10]:

    st.markdown(
        f"### [{article.title}]({article.link})"
    )