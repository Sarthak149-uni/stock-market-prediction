import yfinance as yf

def get_stock_data(ticker, start, end):

    data = yf.download(
        ticker,
        start=start,
        end=end
    )

    return data