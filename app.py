import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh

from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Stock & Crypto — Live Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# PREMIUM DARK THEME CSS
# ======================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --accent-primary: #6C63FF;
        --accent-secondary: #a78bfa;
        --accent-gradient: linear-gradient(135deg, #6C63FF 0%, #a78bfa 50%, #c084fc 100%);
        --green: #10b981;
        --green-glow: rgba(16, 185, 129, 0.3);
        --red: #ef4444;
        --red-glow: rgba(239, 68, 68, 0.3);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== Live Price Ticker ===== */
    .live-ticker {
        background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(17,24,39,0.95) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 1.8rem 2.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .live-ticker::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -15%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(108,99,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .ticker-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        position: relative;
        z-index: 1;
    }
    .ticker-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .ticker-symbol {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .ticker-market-badge {
        background: rgba(108,99,255,0.15);
        color: var(--accent-secondary);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        letter-spacing: 1px;
        border: 1px solid rgba(108,99,255,0.2);
    }
    .ticker-price {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    .ticker-price-up { color: var(--green); text-shadow: 0 0 30px var(--green-glow); }
    .ticker-price-down { color: var(--red); text-shadow: 0 0 30px var(--red-glow); }
    .ticker-change {
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 1rem;
    }
    .ticker-time {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ===== Market Status ===== */
    .market-open {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .market-closed {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .market-247 {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }

    /* ===== Glassmorphism Cards ===== */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
        transform: translateY(-2px);
    }

    /* ===== Metric Cards ===== */
    .metric-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent-gradient);
    }
    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.4);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
        transform: translateY(-3px);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-delta-up { color: var(--green); font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; }
    .metric-delta-down { color: var(--red); font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; }

    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-glass);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-badge {
        background: var(--accent-gradient);
        color: white;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .badge-live {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        animation: glow-pulse 2s ease-in-out infinite;
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 6px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 18px rgba(16, 185, 129, 0.7); }
    }

    /* ===== Prediction Banner ===== */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .prediction-banner.bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(248, 113, 113, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.25);
    }
    .prediction-icon { font-size: 2.2rem; }
    .prediction-text { font-size: 1.2rem; font-weight: 700; color: var(--text-primary); }
    .prediction-sub { font-size: 0.85rem; color: var(--text-secondary); }

    /* ===== News Cards ===== */
    .news-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .news-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        transform: translateX(4px);
    }
    .news-title { color: var(--text-primary); font-weight: 600; font-size: 0.9rem; line-height: 1.4; }
    .news-meta { color: var(--text-muted); font-size: 0.75rem; margin-top: 0.2rem; }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    .sidebar-header {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border-glass);
        margin: 0.8rem 0;
    }
    .rt-indicator {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(52,211,153,0.05) 100%);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        margin: 0.4rem 0 0.8rem;
    }
    .rt-indicator-off {
        background: rgba(100,116,139,0.1);
        border: 1px solid rgba(100,116,139,0.2);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        margin: 0.4rem 0 0.8rem;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: var(--bg-glass);
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid var(--border-glass);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
    }

    /* ===== Compare Table ===== */
    .compare-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-glass);
    }
    .compare-table th {
        background: rgba(108,99,255,0.1);
        color: var(--accent-secondary);
        padding: 0.7rem 1rem;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .compare-table td {
        padding: 0.7rem 1rem;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-glass);
    }
    .compare-table tr:last-child td { border-bottom: none; }

    /* ===== Animations ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .live-dot {
        width: 8px; height: 8px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
        margin-right: 5px;
        box-shadow: 0 0 10px var(--green-glow);
    }
    @keyframes price-flash-up {
        0% { background: transparent; }
        30% { background: rgba(16, 185, 129, 0.15); }
        100% { background: transparent; }
    }
    @keyframes price-flash-down {
        0% { background: transparent; }
        30% { background: rgba(239, 68, 68, 0.15); }
        100% { background: transparent; }
    }
    .flash-up { animation: price-flash-up 1.5s ease-out; }
    .flash-down { animation: price-flash-down 1.5s ease-out; }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

    .footer {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
    }
    .footer-text { color: var(--text-muted); font-size: 0.8rem; }
    .footer-brand {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ======================================================
# MARKET REGISTRY
# ======================================================

MARKETS = {
    "NSE": {
        "icon": "🇮🇳", "label": "NSE India", "suffix": ".NS", "currency": "₹",
        "tz": "Asia/Kolkata", "open": 9, "close": 15, "open_min": 15, "close_min": 30,
        "stocks": {
            "Reliance": "RELIANCE", "TCS": "TCS", "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK", "SBI": "SBIN", "ITC": "ITC",
            "Tata Motors": "TATAMOTORS", "Wipro": "WIPRO",
            "Bajaj Finance": "BAJFINANCE", "Maruti Suzuki": "MARUTI",
            "Kotak Bank": "KOTAKBANK", "L&T": "LT", "HCL Tech": "HCLTECH",
            "Asian Paints": "ASIANPAINT", "Axis Bank": "AXISBANK",
            "Bharti Airtel": "BHARTIARTL", "Sun Pharma": "SUNPHARMA",
            "Titan": "TITAN", "Tata Steel": "TATASTEEL", "Power Grid": "POWERGRID"
        }
    },
    "BSE": {
        "icon": "🇮🇳", "label": "BSE India", "suffix": ".BO", "currency": "₹",
        "tz": "Asia/Kolkata", "open": 9, "close": 15, "open_min": 15, "close_min": 30,
        "stocks": {
            "Reliance": "RELIANCE", "TCS": "TCS", "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK", "SBI": "SBIN", "ITC": "ITC",
            "Tata Motors": "TATAMOTORS", "Wipro": "WIPRO",
            "Bajaj Finance": "BAJFINANCE", "Maruti Suzuki": "MARUTI"
        }
    },
    "US": {
        "icon": "🇺🇸", "label": "US Market", "suffix": "", "currency": "$",
        "tz": "America/New_York", "open": 9, "close": 16, "open_min": 30, "close_min": 0,
        "stocks": {
            "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
            "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META",
            "NVIDIA": "NVDA", "Netflix": "NFLX", "AMD": "AMD",
            "Intel": "INTC", "Berkshire": "BRK-B", "JPMorgan": "JPM",
            "Visa": "V", "Walmart": "WMT", "Disney": "DIS",
            "PayPal": "PYPL", "Uber": "UBER", "Spotify": "SPOT",
            "Snowflake": "SNOW", "Palantir": "PLTR"
        }
    },
    "CRYPTO": {
        "icon": "🪙", "label": "Crypto", "suffix": "-USD", "currency": "$",
        "tz": None, "open": None, "close": None,
        "stocks": {
            "Bitcoin": "BTC", "Ethereum": "ETH", "Solana": "SOL",
            "XRP": "XRP", "Dogecoin": "DOGE", "Cardano": "ADA",
            "Avalanche": "AVAX", "Polkadot": "DOT", "Chainlink": "LINK",
            "Polygon": "MATIC", "Litecoin": "LTC", "Uniswap": "UNI",
            "Shiba Inu": "SHIB", "Stellar": "XLM",
            "Toncoin": "TON11419", "Near Protocol": "NEAR",
            "Sui": "SUI20947", "Aptos": "APT21794",
            "Pepe": "PEPE24478", "Render": "RNDR"
        }
    },
    "INDEX": {
        "icon": "📊", "label": "Indices", "suffix": "", "currency": "",
        "tz": None, "open": None, "close": None,
        "stocks": {
            "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
            "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
            "Russell 2000": "^RUT", "FTSE 100": "^FTSE",
            "DAX": "^GDAXI", "Nikkei 225": "^N225",
            "Hang Seng": "^HSI", "Shanghai": "000001.SS"
        }
    }
}


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def is_market_open(market_key):
    """Check if the market is currently open."""
    info = MARKETS[market_key]
    if market_key == "CRYPTO":
        return "24/7"
    if info.get("tz") is None:
        return "unknown"
    tz = pytz.timezone(info["tz"])
    now = datetime.now(tz)
    if now.weekday() >= 5:  # Saturday/Sunday
        return "closed"
    market_open = now.replace(hour=info["open"], minute=info.get("open_min", 0), second=0)
    market_close = now.replace(hour=info["close"], minute=info.get("close_min", 0), second=0)
    return "open" if market_open <= now <= market_close else "closed"


def get_market_status_html(market_key):
    status = is_market_open(market_key)
    if status == "24/7":
        return '<span class="market-247"><span class="live-dot" style="background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.3);"></span>24/7 OPEN</span>'
    elif status == "open":
        return '<span class="market-open"><span class="live-dot"></span>MARKET OPEN</span>'
    elif status == "closed":
        return '<span class="market-closed">● MARKET CLOSED</span>'
    return ''


def calculate_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(close_series, fast=12, slow=26, signal=9):
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_bollinger(close_series, period=20, std_dev=2):
    sma = close_series.rolling(period).mean()
    std = close_series.rolling(period).std()
    return sma + (std_dev * std), sma, sma - (std_dev * std)


def fmt_price(price, currency):
    if currency == "₹":
        return f"₹{price:,.2f}"
    elif currency == "$":
        return f"${price:,.2f}"
    return f"{price:,.2f}"


def fmt_volume(num, currency=""):
    if currency == "₹":
        if abs(num) >= 1e7: return f"₹{num/1e7:.2f}Cr"
        if abs(num) >= 1e5: return f"₹{num/1e5:.2f}L"
        return f"₹{num:,.0f}"
    if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num/1e6:.2f}M"
    if abs(num) >= 1e3: return f"{num/1e3:.1f}K"
    return f"{num:,.0f}"


def chart_layout(height=500):
    return dict(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(family="Inter", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.markdown('<p class="sidebar-header">⚡ Live Terminal</p>', unsafe_allow_html=True)

    # Real-time controls
    realtime_mode = st.toggle("🔴 Real-Time Mode", value=True)

    if realtime_mode:
        refresh_sec = st.select_slider(
            "Refresh Rate",
            options=[5, 10, 15, 30, 60],
            value=10,
            format_func=lambda x: f"{x}s"
        )
        st.markdown(f'<div class="rt-indicator"><span class="live-dot"></span><span style="color:#10b981;font-weight:600;font-size:0.8rem;">LIVE — Every {refresh_sec}s</span></div>', unsafe_allow_html=True)
        count = st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")
    else:
        st.markdown('<div class="rt-indicator-off"><span style="color:var(--text-muted);font-size:0.8rem;">⏸ Manual Mode</span></div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Market
    market = st.selectbox(
        "🌐 Market",
        list(MARKETS.keys()),
        format_func=lambda x: f"{MARKETS[x]['icon']} {MARKETS[x]['label']}"
    )
    mkt = MARKETS[market]
    currency = mkt["currency"]

    # Asset
    if market == "INDEX":
        asset_name = st.selectbox("📊 Index", list(mkt["stocks"].keys()))
        ticker = mkt["stocks"][asset_name]
    else:
        asset_name = st.selectbox("⭐ Asset", list(mkt["stocks"].keys()))
        custom = st.text_input("🔍 Or type symbol", mkt["stocks"][asset_name])
        ticker = custom.upper() + mkt["suffix"]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Interval
    if realtime_mode:
        rt_interval = st.selectbox(
            "⏱️ Candle Size",
            ["1m", "2m", "5m", "15m", "30m", "1h"],
            index=0,
            format_func=lambda x: {"1m": "1 Min", "2m": "2 Min", "5m": "5 Min", "15m": "15 Min", "30m": "30 Min", "1h": "1 Hour"}[x]
        )
        # yfinance max periods for intraday
        rt_period_map = {"1m": "1d", "2m": "5d", "5m": "5d", "15m": "1mo", "30m": "1mo", "1h": "6mo"}
        interval = rt_interval
        period = rt_period_map[rt_interval]
        use_period = True
    else:
        hist_interval = st.selectbox(
            "⏱️ Interval",
            ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
            index=6,
            format_func=lambda x: {"1m":"1 Min","2m":"2 Min","5m":"5 Min","15m":"15 Min","30m":"30 Min","1h":"1 Hour","1d":"1 Day","1wk":"1 Week","1mo":"1 Month"}[x]
        )
        interval = hist_interval
        if interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
            period = {"1m":"1d","2m":"5d","5m":"5d","15m":"1mo","30m":"1mo","1h":"6mo"}[interval]
            use_period = True
        else:
            start_date = st.date_input("📅 From", pd.to_datetime("2020-01-01"))
            end_date = st.date_input("📅 To", pd.to_datetime("today"))
            use_period = False

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Comparison
    enable_compare = st.checkbox("🔄 Compare", value=False)
    if enable_compare:
        cmp_market = st.selectbox("Compare Market", list(MARKETS.keys()),
            format_func=lambda x: f"{MARKETS[x]['icon']} {MARKETS[x]['label']}", key="cmp_mkt")
        cmp_info = MARKETS[cmp_market]
        cmp_asset = st.selectbox("Compare With", list(cmp_info["stocks"].keys()), key="cmp_ast")
        compare_ticker = cmp_info["stocks"][cmp_asset] + cmp_info["suffix"] if cmp_market != "INDEX" else cmp_info["stocks"][cmp_asset]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    now_str = datetime.now().strftime('%H:%M:%S')
    st.markdown(f'<div style="text-align:center;"><p style="color:var(--text-muted);font-size:0.7rem;">Built by <span class="footer-brand">Sarthak Uniyal</span><br>⏱ {now_str}</p></div>', unsafe_allow_html=True)


# ======================================================
# FETCH DATA (NO CACHE in real-time for truly live data)
# ======================================================

def fetch_live(ticker, interval, period=None, start=None, end=None):
    """Fetch data without caching for real-time freshness."""
    try:
        if period:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        else:
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns and isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_cached(ticker, interval, period=None, start=None, end=None):
    """Cached version for historical/non-realtime data."""
    return fetch_live(ticker, interval, period, start, end)


with st.spinner("⚡ Fetching live market data..."):
    if realtime_mode:
        # No cache — always fresh
        data = fetch_live(ticker, interval, period=period)
    else:
        if use_period:
            data = fetch_cached(ticker, interval, period=period)
        else:
            data = fetch_cached(ticker, interval, start=str(start_date), end=str(end_date))

if data.empty:
    st.error("❌ No data found. Check the symbol and try again.")
    st.stop()


# ======================================================
# COMPUTE LIVE METRICS
# ======================================================

close_arr = np.array(data["Close"]).flatten()
latest_price = float(close_arr[-1])
prev_price = float(close_arr[-2]) if len(close_arr) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0

day_high = float(np.array(data["High"]).flatten()[-1])
day_low = float(np.array(data["Low"]).flatten()[-1])
day_open = float(np.array(data["Open"]).flatten()[0])  # First candle open = session open

session_high = float(np.array(data["High"]).flatten().max())
session_low = float(np.array(data["Low"]).flatten().min())

vol_arr = np.array(data["Volume"]).flatten() if "Volume" in data.columns else None
volume = float(vol_arr.sum()) if vol_arr is not None else 0  # Total session volume
last_vol = float(vol_arr[-1]) if vol_arr is not None and len(vol_arr) > 0 else 0

display_name = ticker.replace(".NS", "").replace(".BO", "").replace("-USD", "")
is_up = price_change >= 0


# ======================================================
# LIVE PRICE TICKER BANNER
# ======================================================

price_class = "ticker-price-up" if is_up else "ticker-price-down"
delta_icon = "▲" if is_up else "▼"
delta_color = "#10b981" if is_up else "#ef4444"
flash_class = "flash-up" if is_up else "flash-down"
market_status_html = get_market_status_html(market)

st.markdown(f"""
<div class="live-ticker {flash_class}">
    <div class="ticker-row">
        <div class="ticker-left">
            <span class="ticker-symbol">{mkt['icon']} {display_name}</span>
            <span class="ticker-market-badge">{mkt['label']}</span>
            {market_status_html}
        </div>
        <div style="display:flex; align-items:baseline; gap: 0.5rem;">
            <span class="ticker-price {price_class}">{fmt_price(latest_price, currency)}</span>
            <span class="ticker-change" style="color:{delta_color}">
                {delta_icon} {abs(price_change):.2f} ({abs(pct_change):.2f}%)
            </span>
        </div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:0.6rem;position:relative;z-index:1;">
        <span class="ticker-time">Open: {fmt_price(day_open, currency)}</span>
        <span class="ticker-time">High: {fmt_price(session_high, currency)}</span>
        <span class="ticker-time">Low: {fmt_price(session_low, currency)}</span>
        <span class="ticker-time">Vol: {fmt_volume(volume, currency)}</span>
        <span class="ticker-time"><span class="live-dot"></span>{datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ======================================================
# MINI METRICS ROW
# ======================================================

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
open_change = ((latest_price - day_open) / day_open) * 100 if day_open != 0 else 0
spread = session_high - session_low
spread_pct = (spread / session_low) * 100 if session_low != 0 else 0

# VWAP (if volume available)
if vol_arr is not None and volume > 0:
    typical = (np.array(data["High"]).flatten() + np.array(data["Low"]).flatten() + close_arr) / 3
    vwap = float(np.sum(typical * vol_arr) / np.sum(vol_arr))
else:
    vwap = latest_price

metrics = [
    ("Open", fmt_price(day_open, currency), None),
    ("High", fmt_price(session_high, currency), None),
    ("Low", fmt_price(session_low, currency), None),
    ("Spread", f"{spread_pct:.2f}%", None),
    ("VWAP", fmt_price(vwap, currency), None),
    ("Candle Vol", fmt_volume(last_vol, currency), None),
    ("Total Vol", fmt_volume(volume, currency), None),
    ("From Open", f"{'+' if open_change>=0 else ''}{open_change:.2f}%",
     "metric-delta-up" if open_change >= 0 else "metric-delta-down"),
]

for col, (label, value, cls) in zip([m1, m2, m3, m4, m5, m6, m7, m8], metrics):
    with col:
        delta_html = f'<div class="{cls}">{value}</div>' if cls else f'<div class="metric-value" style="font-size:1.1rem;">{value}</div>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ======================================================
# MAIN TABS
# ======================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Chart", "📈 Indicators", "🤖 AI Predict",
    "📰 News", "🔄 Compare", "📋 Data"
])


# ======================================================
# TAB 1: LIVE CHART
# ======================================================

with tab1:
    live_badge = '<span class="badge-live"><span class="live-dot"></span>STREAMING</span>' if realtime_mode else '<span class="section-badge">INTERACTIVE</span>'
    st.markdown(f'<div class="section-header">📊 Price Action {live_badge}</div>', unsafe_allow_html=True)

    chart_type = st.radio("Style", ["Candlestick", "Line", "Area"], horizontal=True, key="ct")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="rgba(16,185,129,0.8)",
            decreasing_fillcolor="rgba(239,68,68,0.8)",
            name="Price"
        ), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], mode="lines",
            line=dict(color="#6C63FF", width=2), name="Close"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], mode="lines",
            line=dict(color="#6C63FF", width=2),
            fill="tozeroy", fillcolor="rgba(108,99,255,0.12)", name="Close"
        ), row=1, col=1)

    # VWAP line on chart
    if vol_arr is not None and volume > 0:
        typical_p = (np.array(data["High"]).flatten() + np.array(data["Low"]).flatten() + close_arr) / 3
        cumvol = np.cumsum(vol_arr)
        running_vwap = np.cumsum(typical_p * vol_arr) / np.where(cumvol == 0, 1, cumvol)
        fig.add_trace(go.Scatter(
            x=data.index, y=running_vwap, mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            name="VWAP", opacity=0.7
        ), row=1, col=1)

    # Volume bars
    if "Volume" in data.columns:
        vcolors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(data["Close"], data["Open"])]
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"],
            marker_color=vcolors, opacity=0.5, name="Volume", showlegend=False
        ), row=2, col=1)

    # Latest price horizontal line
    fig.add_hline(y=latest_price, line_dash="dash", line_color="rgba(108,99,255,0.4)",
                  annotation_text=f"  {fmt_price(latest_price, currency)}",
                  annotation_font_color="#a78bfa", row=1, col=1)

    ly = chart_layout(700)
    ly["xaxis_rangeslider_visible"] = False
    ly["yaxis"] = dict(gridcolor="rgba(255,255,255,0.04)", title=f"Price ({currency})")
    ly["yaxis2"] = dict(gridcolor="rgba(255,255,255,0.04)", title="Volume")
    fig.update_layout(**ly)

    # For intraday, remove weekend gaps
    if interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
        fig.update_xaxes(type="category", nticks=20, row=1, col=1)
        fig.update_xaxes(type="category", nticks=20, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# TAB 2: INDICATORS
# ======================================================

with tab2:
    itabs = st.tabs(["Moving Averages", "RSI", "MACD", "Bollinger"])

    with itabs[0]:
        st.markdown('<div class="section-header">📉 Moving Averages <span class="section-badge">MA50•MA100•MA200</span></div>', unsafe_allow_html=True)
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e2e8f0", width=1.5)))
        for ma, col, clr in [(50,"MA50","#6C63FF"), (100,"MA100","#a78bfa"), (200,"MA200","#c084fc")]:
            vals = data["Close"].rolling(ma).mean()
            fig_ma.add_trace(go.Scatter(x=data.index, y=vals, name=col, line=dict(color=clr, width=2)))
        fig_ma.update_layout(**chart_layout(450))
        st.plotly_chart(fig_ma, use_container_width=True)

    with itabs[1]:
        st.markdown('<div class="section-header">📊 RSI <span class="section-badge">14-Period</span></div>', unsafe_allow_html=True)
        rsi = calculate_rsi(data["Close"])
        cur_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50
        rc = "#10b981" if 30 < cur_rsi < 70 else ("#ef4444" if cur_rsi >= 70 else "#3b82f6")
        rs = "Neutral" if 30 < cur_rsi < 70 else ("Overbought" if cur_rsi >= 70 else "Oversold")
        st.markdown(f'<div class="glass-card" style="text-align:center;margin-bottom:1rem;"><span style="font-size:2rem;font-weight:700;color:{rc};">{cur_rsi:.1f}</span> <span style="color:{rc};font-weight:600;">— {rs}</span></div>', unsafe_allow_html=True)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, line=dict(color="#a78bfa", width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="70")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(59,130,246,0.5)", annotation_text="30")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(108,99,255,0.05)", line_width=0)
        l = chart_layout(380)
        l["yaxis"]["range"] = [0, 100]
        fig_rsi.update_layout(**l)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with itabs[2]:
        st.markdown('<div class="section-header">📈 MACD <span class="section-badge">12-26-9</span></div>', unsafe_allow_html=True)
        ml, sl, hist = calculate_macd(data["Close"])
        fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
        fig_m.add_trace(go.Scatter(x=data.index, y=data["Close"], line=dict(color="#e2e8f0", width=1.5), name="Price"), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=data.index, y=ml, line=dict(color="#6C63FF", width=2), name="MACD"), row=2, col=1)
        fig_m.add_trace(go.Scatter(x=data.index, y=sl, line=dict(color="#ef4444", width=1.5), name="Signal"), row=2, col=1)
        hc = ["#10b981" if h >= 0 else "#ef4444" for h in hist]
        fig_m.add_trace(go.Bar(x=data.index, y=hist, marker_color=hc, opacity=0.6, showlegend=False), row=2, col=1)
        lm = chart_layout(550)
        lm["yaxis2"] = dict(gridcolor="rgba(255,255,255,0.04)")
        fig_m.update_layout(**lm)
        st.plotly_chart(fig_m, use_container_width=True)

    with itabs[3]:
        st.markdown('<div class="section-header">📉 Bollinger Bands <span class="section-badge">20-SMA ± 2σ</span></div>', unsafe_allow_html=True)
        upper, middle, lower = calculate_bollinger(data["Close"])
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=data.index, y=upper, name="Upper", line=dict(color="rgba(239,68,68,0.5)", width=1)))
        fig_b.add_trace(go.Scatter(x=data.index, y=lower, name="Lower", line=dict(color="rgba(59,130,246,0.5)", width=1), fill="tonexty", fillcolor="rgba(108,99,255,0.05)"))
        fig_b.add_trace(go.Scatter(x=data.index, y=middle, name="SMA20", line=dict(color="#a78bfa", width=1.5, dash="dot")))
        fig_b.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e2e8f0", width=1.5)))
        fig_b.update_layout(**chart_layout(450))
        st.plotly_chart(fig_b, use_container_width=True)


# ======================================================
# TAB 3: AI PREDICTION
# ======================================================

with tab3:
    st.markdown('<div class="section-header">🤖 AI Prediction <span class="section-badge">LSTM</span></div>', unsafe_allow_html=True)

    try:
        if not TF_AVAILABLE:
            st.warning("⚠️ TensorFlow not installed.")
            st.stop()

        pred_data = fetch_cached(ticker, "1d", start="2015-01-01", end=datetime.now().strftime("%Y-%m-%d"))
        if pred_data.empty or len(pred_data) < 120:
            st.warning("⚠️ Need ≥120 days of daily data for AI prediction.")
            st.stop()

        model = load_model("model/stock_lstm_model.keras")
        close_data = pred_data[["Close"]]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_data)

        x_t, y_t = [], []
        for i in range(100, len(scaled)):
            x_t.append(scaled[i-100:i])
            y_t.append(scaled[i])
        x_t, y_t = np.array(x_t), np.array(y_t)

        with st.spinner("🧠 Running LSTM..."):
            preds = scaler.inverse_transform(model.predict(x_t, verbose=0))
        actuals = scaler.inverse_transform(y_t)
        dates = pred_data.index[-len(actuals):]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=dates, y=actuals.flatten(), name="Actual", line=dict(color="#e2e8f0", width=2)))
        fig_p.add_trace(go.Scatter(x=dates, y=preds.flatten(), name="Predicted", line=dict(color="#6C63FF", width=2)))
        fig_p.update_layout(**chart_layout(450))
        st.plotly_chart(fig_p, use_container_width=True)

        # Next day
        fut = np.reshape(scaled[-100:], (1, 100, 1))
        nd_price = scaler.inverse_transform(model.predict(fut, verbose=0))[0][0]
        bull = nd_price >= latest_price
        chg = nd_price - latest_price
        chg_p = (chg / latest_price) * 100

        st.markdown(f"""
        <div class="prediction-banner{'' if bull else ' bearish'}">
            <span class="prediction-icon">{'🚀' if bull else '📉'}</span>
            <div>
                <div class="prediction-text">Next Day: {fmt_price(nd_price, currency)}</div>
                <div class="prediction-sub">{'BULLISH' if bull else 'BEARISH'} • {currency}{chg:+,.2f} ({chg_p:+.2f}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Forecasts
        st.markdown('<div class="section-header">🔮 Forecast <span class="section-badge">7 & 30 DAY</span></div>', unsafe_allow_html=True)
        def forecast(days):
            fc, inp = [], scaled[-100:].copy()
            for _ in range(days):
                p = model.predict(np.reshape(inp, (1,100,1)), verbose=0)
                fc.append(scaler.inverse_transform(p)[0][0])
                inp = np.append(inp[1:], p, axis=0)
            return fc

        f7, f30 = forecast(7), forecast(30)
        ld = pred_data.index[-1]
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📅 7-Day")
            d7 = pd.date_range(ld + timedelta(1), periods=7, freq='B')
            fg7 = go.Figure()
            fg7.add_trace(go.Scatter(x=pred_data.index[-30:], y=pred_data["Close"].iloc[-30:].values.flatten(), name="Hist", line=dict(color="#e2e8f0", width=1.5)))
            fg7.add_trace(go.Scatter(x=d7, y=f7, name="Forecast", line=dict(color="#10b981", width=2.5, dash="dot"), mode="lines+markers", marker=dict(size=5)))
            fg7.update_layout(**chart_layout(320))
            fg7.update_layout(showlegend=False)
            st.plotly_chart(fg7, use_container_width=True)
            fc7 = (f7[-1] - latest_price) / latest_price * 100
            st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:{"#10b981" if fc7>=0 else "#ef4444"};font-weight:700;font-size:1.1rem;">{fmt_price(f7[-1],currency)} ({"▲" if fc7>=0 else "▼"}{abs(fc7):.2f}%)</span></div>', unsafe_allow_html=True)

        with c2:
            st.markdown("#### 📅 30-Day")
            d30 = pd.date_range(ld + timedelta(1), periods=30, freq='B')
            fg30 = go.Figure()
            fg30.add_trace(go.Scatter(x=pred_data.index[-60:], y=pred_data["Close"].iloc[-60:].values.flatten(), name="Hist", line=dict(color="#e2e8f0", width=1.5)))
            fg30.add_trace(go.Scatter(x=d30, y=f30, name="Forecast", line=dict(color="#c084fc", width=2.5, dash="dot"), mode="lines+markers", marker=dict(size=4)))
            fg30.update_layout(**chart_layout(320))
            fg30.update_layout(showlegend=False)
            st.plotly_chart(fg30, use_container_width=True)
            fc30 = (f30[-1] - latest_price) / latest_price * 100
            st.markdown(f'<div class="glass-card" style="text-align:center;"><span style="color:{"#10b981" if fc30>=0 else "#ef4444"};font-weight:700;font-size:1.1rem;">{fmt_price(f30[-1],currency)} ({"▲" if fc30>=0 else "▼"}{abs(fc30):.2f}%)</span></div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown('<div class="glass-card"><p style="color:var(--text-secondary);">⚠️ Model not loaded. Run <code>python train_model.py</code></p></div>', unsafe_allow_html=True)
        st.error(f"Error: {e}")


# ======================================================
# TAB 4: NEWS
# ======================================================

with tab4:
    st.markdown('<div class="section-header">📰 News <span class="badge-live"><span class="live-dot"></span>LIVE</span></div>', unsafe_allow_html=True)
    try:
        q = f"{asset_name} {'crypto' if market == 'CRYPTO' else 'stock market' if market == 'INDEX' else 'stock'}"
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}")
        if feed.entries:
            for a in feed.entries[:12]:
                pub = getattr(a, 'published', '')
                st.markdown(f'<div class="news-card"><div class="news-title">📄 {a.title}</div><div class="news-meta">{pub}</div></div>', unsafe_allow_html=True)
                st.link_button("Read →", a.link)
        else:
            st.info("No news found.")
    except Exception:
        st.warning("⚠️ Unable to fetch news.")


# ======================================================
# TAB 5: COMPARISON
# ======================================================

with tab5:
    st.markdown('<div class="section-header">🔄 Comparison <span class="section-badge">HEAD-TO-HEAD</span></div>', unsafe_allow_html=True)

    if enable_compare:
        with st.spinner("Loading..."):
            if realtime_mode or use_period:
                cd = fetch_live(compare_ticker, interval, period=period) if realtime_mode else fetch_cached(compare_ticker, interval, period=period)
            else:
                cd = fetch_cached(compare_ticker, interval, start=str(start_date), end=str(end_date))

        if not cd.empty:
            nm = (data["Close"] / data["Close"].iloc[0]) * 100
            nc = (cd["Close"] / cd["Close"].iloc[0]) * 100
            cdn = compare_ticker.replace(".NS","").replace(".BO","").replace("-USD","")

            fc = go.Figure()
            fc.add_trace(go.Scatter(x=data.index, y=nm.values.flatten(), name=display_name, line=dict(color="#6C63FF", width=2.5)))
            fc.add_trace(go.Scatter(x=cd.index, y=nc.values.flatten(), name=cdn, line=dict(color="#10b981", width=2.5)))
            l = chart_layout(450)
            l["yaxis"]["title"] = "Normalized (Base=100)"
            fc.update_layout(**l)
            st.plotly_chart(fc, use_container_width=True)

            cl = float(np.array(cd["Close"]).flatten()[-1])
            cp = float(np.array(cd["Close"]).flatten()[-2]) if len(cd) > 1 else cl
            cpct = ((cl - cp) / cp) * 100 if cp != 0 else 0
            cc = MARKETS[cmp_market]["currency"]

            st.markdown(f"""
            <table class="compare-table">
                <tr><th>Metric</th><th>{display_name}</th><th>{cdn}</th></tr>
                <tr><td>Price</td><td>{fmt_price(latest_price, currency)}</td><td>{fmt_price(cl, cc)}</td></tr>
                <tr><td>Change</td><td style="color:{'#10b981' if pct_change>=0 else '#ef4444'}">{pct_change:+.2f}%</td><td style="color:{'#10b981' if cpct>=0 else '#ef4444'}">{cpct:+.2f}%</td></tr>
            </table>""", unsafe_allow_html=True)
        else:
            st.error("Could not fetch comparison data.")
    else:
        st.info("💡 Enable **Compare** in sidebar.")


# ======================================================
# TAB 6: DATA
# ======================================================

with tab6:
    st.markdown('<div class="section-header">📋 Data <span class="section-badge">RAW</span></div>', unsafe_allow_html=True)
    dd = data.copy()
    try:
        dd.index = dd.index.strftime('%Y-%m-%d %H:%M')
    except Exception:
        pass
    rows = st.slider("Rows", 10, min(500, len(data)), 50)
    st.dataframe(dd.tail(rows), use_container_width=True, height=450)
    st.download_button("📥 CSV", data.to_csv(), f"{display_name}_live.csv", "text/csv")


# ======================================================
# FOOTER
# ======================================================

st.markdown(f"""
<div class="footer">
    <p class="footer-text">
        Built by <span class="footer-brand">Sarthak Uniyal</span> •
        <span class="footer-brand">LSTM AI</span> •
        <span class="footer-brand">Yahoo Finance</span> •
        ⚠️ Educational only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)