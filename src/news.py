import feedparser

def get_news(stock):

    url = (
        f"https://news.google.com/rss/search?q="
        f"{stock}+stock"
    )

    return feedparser.parse(url)