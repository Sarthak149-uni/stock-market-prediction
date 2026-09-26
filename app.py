import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timedelta
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
    page_title="AI Stock Market Predictor",
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

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --accent-primary: #6C63FF;
        --accent-secondary: #a78bfa;
        --accent-gradient: linear-gradient(135deg, #6C63FF 0%, #a78bfa 50%, #c084fc 100%);
        --green: #10b981;
        --green-glow: rgba(16, 185, 129, 0.2);
        --red: #ef4444;
        --red-glow: rgba(239, 68, 68, 0.2);
        --yellow: #f59e0b;
        --blue: #3b82f6;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== Hero Header ===== */
    .hero-header {
        background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(167,139,250,0.08) 50%, rgba(192,132,252,0.05) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
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
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
        border-radius: 0 0 16px 16px;
    }
    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.4);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.15);
        transform: translateY(-4px);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-delta-up {
        color: var(--green);
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .metric-delta-down {
        color: var(--red);
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-glass);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-badge {
        background: var(--accent-gradient);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .section-badge-live {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
        animation: glow-pulse 2s ease-in-out infinite;
    }
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.7); }
    }

    /* ===== Prediction Banner ===== */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .prediction-banner.bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(248, 113, 113, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.25);
    }
    .prediction-icon { font-size: 2.5rem; }
    .prediction-text { font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
    .prediction-sub { font-size: 0.9rem; color: var(--text-secondary); }

    /* ===== News Cards ===== */
    .news-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0;
        transition: all 0.3s ease;
    }
    .news-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        transform: translateX(4px);
    }
    .news-title { color: var(--text-primary); font-weight: 600; font-size: 0.95rem; line-height: 1.4; }
    .news-meta { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.3rem; }

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
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border-glass);
        margin: 1rem 0;
    }

    /* ===== Real-time badge ===== */
    .realtime-status {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin: 0.5rem 0 1rem;
    }
    .realtime-status-off {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.15) 0%, rgba(100, 116, 139, 0.05) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin: 0.5rem 0 1rem;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-glass);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-glass);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
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
        background: rgba(108, 99, 255, 0.1);
        color: var(--accent-secondary);
        padding: 0.8rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .compare-table td {
        padding: 0.8rem 1rem;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-glass);
        font-size: 0.95rem;
    }
    .compare-table tr:last-child td { border-bottom: none; }

    /* ===== Animations ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
        margin-right: 6px;
        box-shadow: 0 0 8px var(--green-glow);
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

    /* ===== Footer ===== */
    .footer {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .footer-text { color: var(--text-muted); font-size: 0.85rem; }
    .footer-brand {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ===== Hide defaults ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ======================================================
# MARKET DATA REGISTRY
# ======================================================

MARKETS = {
    "NSE": {
        "icon": "🇮🇳",
        "label": "NSE (India)",
        "suffix": ".NS",
        "currency": "₹",
        "stocks": {
            "Reliance": "RELIANCE",
            "TCS": "TCS",
            "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK",
            "SBI": "SBIN",
            "ITC": "ITC",
            "Tata Motors": "TATAMOTORS",
            "Wipro": "WIPRO",
            "Bajaj Finance": "BAJFINANCE",
            "Maruti Suzuki": "MARUTI",
            "Kotak Bank": "KOTAKBANK",
            "L&T": "LT",
            "HCL Tech": "HCLTECH",
            "Asian Paints": "ASIANPAINT",
            "Axis Bank": "AXISBANK",
            "Bharti Airtel": "BHARTIARTL",
            "Sun Pharma": "SUNPHARMA",
            "Titan": "TITAN",
            "Tata Steel": "TATASTEEL",
            "Power Grid": "POWERGRID"
        }
    },
    "BSE": {
        "icon": "🇮🇳",
        "label": "BSE (India)",
        "suffix": ".BO",
        "currency": "₹",
        "stocks": {
            "Reliance": "RELIANCE",
            "TCS": "TCS",
            "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK",
            "SBI": "SBIN",
            "ITC": "ITC",
            "Tata Motors": "TATAMOTORS",
            "Wipro": "WIPRO",
            "Bajaj Finance": "BAJFINANCE",
            "Maruti Suzuki": "MARUTI"
        }
    },
    "US": {
        "icon": "🇺🇸",
        "label": "US Market",
        "suffix": "",
        "currency": "$",
        "stocks": {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Google": "GOOGL",
            "Amazon": "AMZN",
            "Tesla": "TSLA",
            "Meta": "META",
            "NVIDIA": "NVDA",
            "Netflix": "NFLX",
            "AMD": "AMD",
            "Intel": "INTC",
            "Berkshire": "BRK-B",
            "JPMorgan": "JPM",
            "Visa": "V",
            "Walmart": "WMT",
            "Disney": "DIS",
            "PayPal": "PYPL",
            "Uber": "UBER",
            "Spotify": "SPOT",
            "Snowflake": "SNOW",
            "Palantir": "PLTR"
        }
    },
    "CRYPTO": {
        "icon": "🪙",
        "label": "Crypto",
        "suffix": "-USD",
        "currency": "$",
        "stocks": {
            "Bitcoin": "BTC",
            "Ethereum": "ETH",
            "Solana": "SOL",
            "XRP": "XRP",
            "Dogecoin": "DOGE",
            "Cardano": "ADA",
            "Avalanche": "AVAX",
            "Polkadot": "DOT",
            "Chainlink": "LINK",
            "Polygon": "MATIC",
            "Litecoin": "LTC",
            "Uniswap": "UNI",
            "Shiba Inu": "SHIB",
            "Stellar": "XLM",
            "Toncoin": "TON11419",
            "Near Protocol": "NEAR",
            "Sui": "SUI20947",
            "Aptos": "APT21794",
            "Pepe": "PEPE24478",
            "Render": "RNDR"
        }
    },
    "INDEX": {
        "icon": "📊",
        "label": "Indices",
        "suffix": "",
        "currency": "",
        "stocks": {
            "NIFTY 50": "^NSEI",
            "BANK NIFTY": "^NSEBANK",
            "SENSEX": "^BSESN",
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
            "FTSE 100": "^FTSE",
            "DAX": "^GDAXI",
            "Nikkei 225": "^N225",
            "Hang Seng": "^HSI",
            "Shanghai": "000001.SS"
        }
    }
}

INTERVALS = {
    "1 Minute": "1m",
    "2 Minutes": "2m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}

INTRADAY_PERIODS = {
    "1 Minute": "1d",
    "2 Minutes": "5d",
    "5 Minutes": "5d",
    "15 Minutes": "1mo",
    "30 Minutes": "1mo",
    "1 Hour": "6mo"
}


# ======================================================
# HELPER FUNCTIONS
# ======================================================

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
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(close_series, period=20, std_dev=2):
    sma = close_series.rolling(period).mean()
    std = close_series.rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


def get_currency_symbol(market):
    return MARKETS.get(market, {}).get("currency", "")


def format_price(price, currency):
    if currency == "₹":
        return f"₹{price:,.2f}"
    elif currency == "$":
        return f"${price:,.2f}"
    else:
        return f"{price:,.2f}"


def format_large_number(num, currency=""):
    if currency == "₹":
        if abs(num) >= 1e7:
            return f"₹{num/1e7:.2f} Cr"
        elif abs(num) >= 1e5:
            return f"₹{num/1e5:.2f} L"
        else:
            return f"₹{num:,.2f}"
    elif currency == "$":
        if abs(num) >= 1e9:
            return f"${num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:.1f}K"
        else:
            return f"${num:,.2f}"
    else:
        return f"{num:,.0f}"


def get_chart_layout(height=500, title=None):
    layout = dict(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30 if not title else 50, b=0),
        font=dict(family="Inter", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
        )
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=16))
    return layout


# ======================================================
# HERO HEADER
# ======================================================

st.markdown("""
<div class="hero-header">
    <p class="hero-title">📈 AI Stock & Crypto Predictor</p>
    <p class="hero-subtitle">
        <span class="live-dot"></span>
        Real-Time Market Data • NSE • BSE • US Stocks • Crypto • Global Indices • LSTM Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.markdown('<p class="sidebar-header">⚡ Market Terminal</p>', unsafe_allow_html=True)

    # --- Real-Time Mode ---
    st.markdown('<p class="sidebar-header" style="font-size: 1rem;">🔴 Real-Time Mode</p>', unsafe_allow_html=True)

    realtime_mode = st.toggle("Enable Real-Time", value=False, help="Auto-refresh data every 30 seconds")

    if realtime_mode:
        refresh_rate = st.select_slider(
            "Refresh Interval",
            options=[15, 30, 60, 120, 300],
            value=30,
            format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m"
        )
        st.markdown(f"""
        <div class="realtime-status">
            <span class="live-dot"></span>
            <span style="color: #10b981; font-weight: 600; font-size: 0.85rem;">LIVE — Refreshing every {refresh_rate}s</span>
        </div>
        """, unsafe_allow_html=True)
        # Auto-refresh
        st_autorefresh(interval=refresh_rate * 1000, key="data_refresh")
    else:
        st.markdown("""
        <div class="realtime-status-off">
            <span style="color: var(--text-muted); font-size: 0.85rem;">⏸️ Manual mode — pull to refresh</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Market Selection ---
    market = st.selectbox(
        "🌐 Choose Market",
        list(MARKETS.keys()),
        format_func=lambda x: f"{MARKETS[x]['icon']} {MARKETS[x]['label']}"
    )

    market_info = MARKETS[market]
    currency = market_info["currency"]

    # --- Stock/Asset Selection ---
    if market == "INDEX":
        asset_name = st.selectbox(
            "📊 Choose Index",
            list(market_info["stocks"].keys())
        )
        ticker = market_info["stocks"][asset_name]
    elif market == "CRYPTO":
        asset_name = st.selectbox(
            "🪙 Choose Crypto",
            list(market_info["stocks"].keys())
        )
        custom_symbol = st.text_input(
            "🔍 Or Enter Crypto Symbol",
            market_info["stocks"][asset_name]
        )
        ticker = custom_symbol.upper() + market_info["suffix"]
    else:
        asset_name = st.selectbox(
            "⭐ Popular Stocks",
            list(market_info["stocks"].keys())
        )
        custom_symbol = st.text_input(
            "🔍 Or Enter Symbol",
            market_info["stocks"][asset_name]
        )
        ticker = custom_symbol.upper() + market_info["suffix"]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Data Interval ---
    if realtime_mode:
        interval_name = st.selectbox(
            "⏱️ Chart Interval",
            ["1 Minute", "2 Minutes", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"],
            index=2
        )
        interval = INTERVALS[interval_name]
        period = INTRADAY_PERIODS.get(interval_name, "5d")
    else:
        interval_name = st.selectbox(
            "⏱️ Chart Interval",
            list(INTERVALS.keys()),
            index=6  # Default: 1 Day
        )
        interval = INTERVALS[interval_name]

        if interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
            period = INTRADAY_PERIODS.get(interval_name, "5d")
        else:
            start_date = st.date_input("📅 Start Date", pd.to_datetime("2020-01-01"))
            end_date = st.date_input("📅 End Date", pd.to_datetime("today"))
            period = None  # Will use start/end dates

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Stock Comparison ---
    st.markdown('<p class="sidebar-header" style="font-size: 1rem;">🔄 Compare</p>', unsafe_allow_html=True)
    enable_compare = st.checkbox("Enable Comparison", value=False)

    if enable_compare:
        compare_market = st.selectbox(
            "Compare Market",
            list(MARKETS.keys()),
            format_func=lambda x: f"{MARKETS[x]['icon']} {MARKETS[x]['label']}",
            key="compare_market"
        )
        compare_info = MARKETS[compare_market]
        compare_asset = st.selectbox(
            "Compare With",
            list(compare_info["stocks"].keys()),
            key="compare_asset"
        )
        if compare_market == "INDEX":
            compare_ticker = compare_info["stocks"][compare_asset]
        else:
            compare_ticker = compare_info["stocks"][compare_asset] + compare_info["suffix"]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: var(--text-muted); font-size: 0.75rem;">
            Built with ❤️ by <span class="footer-brand">Sarthak Uniyal</span><br>
            Last refresh: {datetime.now().strftime('%H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# DOWNLOAD DATA
# ======================================================

@st.cache_data(ttl=30 if realtime_mode else 300)
def fetch_data(ticker, interval, period=None, start=None, end=None):
    """Fetch stock/crypto data with caching."""
    try:
        if period:
            data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        else:
            data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                if isinstance(data[col], pd.DataFrame):
                    data[col] = data[col].iloc[:, 0]
        return data
    except Exception as e:
        return pd.DataFrame()


with st.spinner("⚡ Fetching live market data..."):
    if interval in ["1m", "2m", "5m", "15m", "30m", "1h"] or period:
        data = fetch_data(ticker, interval, period=period)
    else:
        data = fetch_data(ticker, interval, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found. Please check the symbol and try again.")
    st.stop()


# ======================================================
# COMPUTE METRICS
# ======================================================

close_arr = np.array(data["Close"]).flatten()
latest_price = float(close_arr[-1])
prev_price = float(close_arr[-2]) if len(close_arr) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0

day_high = float(np.array(data["High"]).flatten()[-1])
day_low = float(np.array(data["Low"]).flatten()[-1])
day_open = float(np.array(data["Open"]).flatten()[-1])

vol_arr = np.array(data["Volume"]).flatten() if "Volume" in data.columns else None
volume = float(vol_arr[-1]) if vol_arr is not None and len(vol_arr) > 0 else 0

# Session high/low (all data in current view)
session_high = float(np.array(data["High"]).flatten().max())
session_low = float(np.array(data["Low"]).flatten().min())

# Display name
display_name = ticker.replace(".NS", "").replace(".BO", "").replace("-USD", "")
delta_class = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
delta_icon = "▲" if price_change >= 0 else "▼"
market_icon = MARKETS[market]["icon"]

# ======================================================
# STOCK INFO BAR
# ======================================================

rt_badge = '<span class="section-badge-live"><span class="live-dot"></span> LIVE</span>' if realtime_mode else '<span class="section-badge">DELAYED</span>'

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
    <span style="font-size: 1.5rem; font-weight: 800; color: var(--text-primary);">{market_icon} {display_name}</span>
    {rt_badge}
    <span style="color: var(--text-muted); font-size: 0.9rem;">{MARKETS[market]['label']} • {interval_name}</span>
    <span style="color: var(--text-muted); font-size: 0.8rem;">Updated: {datetime.now().strftime('%H:%M:%S')}</span>
</div>
""", unsafe_allow_html=True)

# Metric cards
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">{format_price(latest_price, currency)}</div>
        <div class="{delta_class}">{delta_icon} {abs(price_change):.2f} ({abs(pct_change):.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Open</div>
        <div class="metric-value">{format_price(day_open, currency)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High</div>
        <div class="metric-value">{format_price(day_high, currency)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Low</div>
        <div class="metric-value">{format_price(day_low, currency)}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Session High</div>
        <div class="metric-value">{format_price(session_high, currency)}</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Volume</div>
        <div class="metric-value">{format_large_number(volume, currency) if volume > 0 else 'N/A'}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ======================================================
# MAIN TABS
# ======================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Charts",
    "📈 Indicators",
    "🤖 AI Prediction",
    "📰 News",
    "🔄 Comparison",
    "📋 Data"
])


# ======================================================
# TAB 1: CHARTS
# ======================================================

with tab1:
    rt_label = '<span class="section-badge-live"><span class="live-dot"></span> REAL-TIME</span>' if realtime_mode else '<span class="section-badge">INTERACTIVE</span>'
    st.markdown(f"""
    <div class="section-header">
        📊 Price Chart {rt_label}
    </div>
    """, unsafe_allow_html=True)

    chart_type = st.radio("Chart Type", ["Candlestick", "Line", "Area"], horizontal=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25]
    )

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
            fill="tozeroy", fillcolor="rgba(108,99,255,0.15)", name="Close"
        ), row=1, col=1)

    if "Volume" in data.columns:
        colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(data["Close"], data["Open"])]
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"],
            marker_color=colors, opacity=0.5, name="Volume", showlegend=False
        ), row=2, col=1)

    layout = get_chart_layout(700)
    layout["xaxis_rangeslider_visible"] = False
    layout["yaxis"] = dict(gridcolor="rgba(255,255,255,0.04)", title=f"Price ({currency})")
    layout["yaxis2"] = dict(gridcolor="rgba(255,255,255,0.04)", title="Volume")
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# TAB 2: INDICATORS
# ======================================================

with tab2:
    ind_tabs = st.tabs(["Moving Averages", "RSI", "MACD", "Bollinger Bands"])

    with ind_tabs[0]:
        st.markdown("""<div class="section-header">📉 Moving Averages <span class="section-badge">MA50 • MA100 • MA200</span></div>""", unsafe_allow_html=True)

        ma50 = data["Close"].rolling(50).mean()
        ma100 = data["Close"].rolling(100).mean()
        ma200 = data["Close"].rolling(200).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e2e8f0", width=1.5)))
        fig_ma.add_trace(go.Scatter(x=data.index, y=ma50, name="MA50", line=dict(color="#6C63FF", width=2)))
        fig_ma.add_trace(go.Scatter(x=data.index, y=ma100, name="MA100", line=dict(color="#a78bfa", width=2)))
        fig_ma.add_trace(go.Scatter(x=data.index, y=ma200, name="MA200", line=dict(color="#c084fc", width=2, dash="dot")))
        fig_ma.update_layout(**get_chart_layout(500))
        st.plotly_chart(fig_ma, use_container_width=True)

    with ind_tabs[1]:
        st.markdown("""<div class="section-header">📊 Relative Strength Index <span class="section-badge">RSI-14</span></div>""", unsafe_allow_html=True)

        rsi = calculate_rsi(data["Close"])
        current_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50
        rsi_color = "#10b981" if 30 < current_rsi < 70 else ("#ef4444" if current_rsi >= 70 else "#3b82f6")
        rsi_status = "Neutral" if 30 < current_rsi < 70 else ("Overbought" if current_rsi >= 70 else "Oversold")

        st.markdown(f"""
        <div class="glass-card" style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; font-weight: 700; color: {rsi_color};">{current_rsi:.1f}</span>
            <span style="color: {rsi_color}; font-weight: 600;"> — {rsi_status}</span>
        </div>
        """, unsafe_allow_html=True)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, line=dict(color="#a78bfa", width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(59,130,246,0.5)", annotation_text="Oversold (30)")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(108,99,255,0.05)", line_width=0)
        layout_rsi = get_chart_layout(400)
        layout_rsi["yaxis"]["range"] = [0, 100]
        fig_rsi.update_layout(**layout_rsi)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with ind_tabs[2]:
        st.markdown("""<div class="section-header">📈 MACD <span class="section-badge">12-26-9</span></div>""", unsafe_allow_html=True)

        macd_line, signal_line, histogram = calculate_macd(data["Close"])

        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["Close"], line=dict(color="#e2e8f0", width=1.5), name="Price"), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=data.index, y=macd_line, line=dict(color="#6C63FF", width=2), name="MACD"), row=2, col=1)
        fig_macd.add_trace(go.Scatter(x=data.index, y=signal_line, line=dict(color="#ef4444", width=1.5), name="Signal"), row=2, col=1)

        hist_colors = ["#10b981" if h >= 0 else "#ef4444" for h in histogram]
        fig_macd.add_trace(go.Bar(x=data.index, y=histogram, marker_color=hist_colors, opacity=0.6, name="Histogram", showlegend=False), row=2, col=1)

        layout_macd = get_chart_layout(600)
        layout_macd["yaxis2"] = dict(gridcolor="rgba(255,255,255,0.04)")
        fig_macd.update_layout(**layout_macd)
        st.plotly_chart(fig_macd, use_container_width=True)

    with ind_tabs[3]:
        st.markdown("""<div class="section-header">📉 Bollinger Bands <span class="section-badge">20-Day SMA ± 2σ</span></div>""", unsafe_allow_html=True)

        upper, middle, lower = calculate_bollinger(data["Close"])

        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index, y=upper, name="Upper", line=dict(color="rgba(239,68,68,0.5)", width=1)))
        fig_bb.add_trace(go.Scatter(x=data.index, y=lower, name="Lower", line=dict(color="rgba(59,130,246,0.5)", width=1), fill="tonexty", fillcolor="rgba(108,99,255,0.05)"))
        fig_bb.add_trace(go.Scatter(x=data.index, y=middle, name="SMA 20", line=dict(color="#a78bfa", width=1.5, dash="dot")))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e2e8f0", width=1.5)))
        fig_bb.update_layout(**get_chart_layout(500))
        st.plotly_chart(fig_bb, use_container_width=True)


# ======================================================
# TAB 3: AI PREDICTION
# ======================================================

with tab3:
    st.markdown("""<div class="section-header">🤖 AI Prediction Engine <span class="section-badge">LSTM DEEP LEARNING</span></div>""", unsafe_allow_html=True)

    try:
        if not TF_AVAILABLE:
            st.warning("⚠️ TensorFlow not installed. Install with: `pip install tensorflow`")
            st.stop()

        # Need daily data for predictions
        pred_data = fetch_data(ticker, "1d", start="2015-01-01", end=datetime.now().strftime("%Y-%m-%d"))

        if pred_data.empty or len(pred_data) < 120:
            st.warning("⚠️ Need at least 120 days of historical data for AI prediction.")
            st.stop()

        model = load_model("model/stock_lstm_model.keras")

        close_data = pred_data[["Close"]]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        x_test, y_test = [], []
        for i in range(100, len(scaled_data)):
            x_test.append(scaled_data[i - 100:i])
            y_test.append(scaled_data[i])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        with st.spinner("🧠 Running LSTM model inference..."):
            predictions = model.predict(x_test, verbose=0)

        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test)

        dates = pred_data.index[-len(actual_prices):]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=dates, y=actual_prices.flatten(), name="Actual", line=dict(color="#e2e8f0", width=2)))
        fig_pred.add_trace(go.Scatter(x=dates, y=predictions.flatten(), name="Predicted", line=dict(color="#6C63FF", width=2)))
        fig_pred.update_layout(**get_chart_layout(500))
        st.plotly_chart(fig_pred, use_container_width=True)

        # Next-Day
        last_100 = scaled_data[-100:]
        future_input = np.reshape(last_100, (1, 100, 1))
        next_day = model.predict(future_input, verbose=0)
        next_day_price = scaler.inverse_transform(next_day)[0][0]

        is_bullish = next_day_price >= latest_price
        banner_class = "" if is_bullish else " bearish"
        banner_icon = "🚀" if is_bullish else "📉"
        trend_text = "BULLISH" if is_bullish else "BEARISH"
        change_val = next_day_price - latest_price
        change_pct = (change_val / latest_price) * 100

        st.markdown(f"""
        <div class="prediction-banner{banner_class}">
            <span class="prediction-icon">{banner_icon}</span>
            <div>
                <div class="prediction-text">Next Day: {format_price(next_day_price, currency)}</div>
                <div class="prediction-sub">{trend_text} • {currency}{change_val:+,.2f} ({change_pct:+.2f}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-step forecasts
        st.markdown("""<div class="section-header">🔮 Multi-Step Forecast <span class="section-badge">7 & 30 DAY</span></div>""", unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)

        def run_forecast(days):
            forecast = []
            inp = scaled_data[-100:].copy()
            for _ in range(days):
                x = np.reshape(inp, (1, 100, 1))
                p = model.predict(x, verbose=0)
                forecast.append(scaler.inverse_transform(p)[0][0])
                inp = np.append(inp[1:], p, axis=0)
            return forecast

        forecast_7 = run_forecast(7)
        forecast_30 = run_forecast(30)

        last_date = pred_data.index[-1]

        with fc1:
            st.markdown("#### 📅 7-Day Forecast")
            fd7 = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='B')
            fig_f7 = go.Figure()
            fig_f7.add_trace(go.Scatter(x=pred_data.index[-30:], y=pred_data["Close"].iloc[-30:].values.flatten(), name="History", line=dict(color="#e2e8f0", width=1.5)))
            fig_f7.add_trace(go.Scatter(x=fd7, y=forecast_7, name="Forecast", line=dict(color="#10b981", width=2.5, dash="dot"), mode="lines+markers", marker=dict(size=6)))
            fig_f7.update_layout(**get_chart_layout(350))
            fig_f7.update_layout(showlegend=False)
            st.plotly_chart(fig_f7, use_container_width=True)

            f7c = forecast_7[-1] - latest_price
            f7p = (f7c / latest_price) * 100
            st.markdown(f"""<div class="glass-card" style="text-align:center;"><span style="color:{'#10b981' if f7c>=0 else '#ef4444'};font-weight:700;font-size:1.2rem;">{format_price(forecast_7[-1], currency)} ({'▲' if f7c>=0 else '▼'} {abs(f7p):.2f}%)</span></div>""", unsafe_allow_html=True)

        with fc2:
            st.markdown("#### 📅 30-Day Forecast")
            fd30 = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='B')
            fig_f30 = go.Figure()
            fig_f30.add_trace(go.Scatter(x=pred_data.index[-60:], y=pred_data["Close"].iloc[-60:].values.flatten(), name="History", line=dict(color="#e2e8f0", width=1.5)))
            fig_f30.add_trace(go.Scatter(x=fd30, y=forecast_30, name="Forecast", line=dict(color="#c084fc", width=2.5, dash="dot"), mode="lines+markers", marker=dict(size=4)))
            fig_f30.update_layout(**get_chart_layout(350))
            fig_f30.update_layout(showlegend=False)
            st.plotly_chart(fig_f30, use_container_width=True)

            f30c = forecast_30[-1] - latest_price
            f30p = (f30c / latest_price) * 100
            st.markdown(f"""<div class="glass-card" style="text-align:center;"><span style="color:{'#10b981' if f30c>=0 else '#ef4444'};font-weight:700;font-size:1.2rem;">{format_price(forecast_30[-1], currency)} ({'▲' if f30c>=0 else '▼'} {abs(f30p):.2f}%)</span></div>""", unsafe_allow_html=True)

    except Exception as e:
        st.markdown("""
        <div class="glass-card">
            <p style="color: var(--text-secondary);">
                ⚠️ <strong>Model not loaded.</strong> Place your trained LSTM model at
                <code>model/stock_lstm_model.keras</code>
            </p>
            <p style="color: var(--text-muted); font-size: 0.85rem;">
                Run <code>python train_model.py</code> to train the model.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Error: {e}")


# ======================================================
# TAB 4: NEWS
# ======================================================

with tab4:
    st.markdown(f"""
    <div class="section-header">
        📰 Market News <span class="section-badge-live"><span class="live-dot"></span> LIVE FEED</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        search_term = display_name
        if market == "CRYPTO":
            search_term = asset_name + " crypto"
        elif market == "INDEX":
            search_term = asset_name + " stock market"
        else:
            search_term = display_name + " stock"

        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={search_term}"
        )

        if feed.entries:
            for article in feed.entries[:12]:
                published = getattr(article, 'published', '')
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">📄 {article.title}</div>
                    <div class="news-meta">{published}</div>
                </div>
                """, unsafe_allow_html=True)
                st.link_button("Read Article →", article.link, use_container_width=False)
        else:
            st.info("No news articles found.")
    except Exception:
        st.warning("⚠️ Unable to fetch news. Check your internet connection.")


# ======================================================
# TAB 5: COMPARISON
# ======================================================

with tab5:
    st.markdown("""<div class="section-header">🔄 Stock Comparison <span class="section-badge">HEAD-TO-HEAD</span></div>""", unsafe_allow_html=True)

    if enable_compare:
        with st.spinner("Fetching comparison data..."):
            if interval in ["1m", "2m", "5m", "15m", "30m", "1h"] or period:
                compare_data = fetch_data(compare_ticker, interval, period=period)
            else:
                compare_data = fetch_data(compare_ticker, interval, start=start_date, end=end_date)

        if not compare_data.empty:
            norm_main = (data["Close"] / data["Close"].iloc[0]) * 100
            norm_comp = (compare_data["Close"] / compare_data["Close"].iloc[0]) * 100

            compare_display = compare_ticker.replace(".NS", "").replace(".BO", "").replace("-USD", "")
            comp_currency = MARKETS[compare_market]["currency"]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=data.index, y=norm_main.values.flatten(), name=display_name, line=dict(color="#6C63FF", width=2.5)))
            fig_cmp.add_trace(go.Scatter(x=compare_data.index, y=norm_comp.values.flatten(), name=compare_display, line=dict(color="#10b981", width=2.5)))
            layout_cmp = get_chart_layout(500)
            layout_cmp["yaxis"]["title"] = "Normalized Price (Base=100)"
            fig_cmp.update_layout(**layout_cmp)
            st.plotly_chart(fig_cmp, use_container_width=True)

            c_latest = float(np.array(compare_data["Close"]).flatten()[-1])
            c_prev = float(np.array(compare_data["Close"]).flatten()[-2]) if len(compare_data) > 1 else c_latest
            c_pct = ((c_latest - c_prev) / c_prev) * 100 if c_prev != 0 else 0
            c_high = float(np.array(compare_data["High"]).flatten()[-1])
            c_low = float(np.array(compare_data["Low"]).flatten()[-1])

            st.markdown(f"""
            <table class="compare-table">
                <tr><th>Metric</th><th>{display_name}</th><th>{compare_display}</th></tr>
                <tr><td>Price</td><td>{format_price(latest_price, currency)}</td><td>{format_price(c_latest, comp_currency)}</td></tr>
                <tr><td>Change</td>
                    <td style="color:{'#10b981' if pct_change>=0 else '#ef4444'}">{pct_change:+.2f}%</td>
                    <td style="color:{'#10b981' if c_pct>=0 else '#ef4444'}">{c_pct:+.2f}%</td>
                </tr>
                <tr><td>High</td><td>{format_price(day_high, currency)}</td><td>{format_price(c_high, comp_currency)}</td></tr>
                <tr><td>Low</td><td>{format_price(day_low, currency)}</td><td>{format_price(c_low, comp_currency)}</td></tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not fetch comparison data.")
    else:
        st.info("💡 Enable **Stock Comparison** in the sidebar to compare two assets head-to-head.")


# ======================================================
# TAB 6: RAW DATA
# ======================================================

with tab6:
    st.markdown("""<div class="section-header">📋 Historical Data <span class="section-badge">RAW</span></div>""", unsafe_allow_html=True)

    data_display = data.copy()
    if hasattr(data_display.index, 'strftime'):
        try:
            data_display.index = data_display.index.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass

    show_rows = st.slider("Rows to display", 10, min(500, len(data)), 50)
    st.dataframe(data_display.tail(show_rows), use_container_width=True, height=500)

    csv = data.to_csv()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{display_name}_data.csv",
        mime="text/csv"
    )


# ======================================================
# FOOTER
# ======================================================

st.markdown(f"""
<div class="footer">
    <p class="footer-text">
        Built with ❤️ by <span class="footer-brand">Sarthak Uniyal</span> •
        Powered by <span class="footer-brand">LSTM Deep Learning</span> •
        Data from <span class="footer-brand">Yahoo Finance</span>
    </p>
    <p class="footer-text" style="margin-top: 0.5rem; font-size: 0.75rem;">
        ⚠️ Educational purposes only. Not financial advice. Crypto markets are highly volatile.
    </p>
</div>
""", unsafe_allow_html=True)