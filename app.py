import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from datetime import datetime, timedelta

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
    /* ===== Google Font ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== Root Variables ===== */
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
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* ===== Base Styles ===== */
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
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-delta-up {
        color: var(--green);
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .metric-delta-down {
        color: var(--red);
        font-size: 0.9rem;
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
    .prediction-icon {
        font-size: 2.5rem;
    }
    .prediction-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .prediction-sub {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

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
    .news-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .news-meta {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* ===== Sidebar Styling ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stDateInput label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
    }

    .sidebar-header {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }

    /* ===== Tab Styling ===== */
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
        padding: 0.5rem 1.5rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

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
    .footer-text {
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .footer-brand {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ===== Comparison Table ===== */
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
    .compare-table tr:last-child td {
        border-bottom: none;
    }

    /* ===== Pulse Animation ===== */
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
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-glass);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }

    /* ===== Hide default streamlit elements ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Dataframe styling ===== */
    .stDataFrame {
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def calculate_rsi(data, period=14):
    """Calculate RSI indicator."""
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(data, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = data["Close"].rolling(period).mean()
    std = data["Close"].rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


def format_large_number(num):
    """Format large numbers for display."""
    if abs(num) >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    elif abs(num) >= 1e5:
        return f"₹{num/1e5:.2f} L"
    else:
        return f"₹{num:,.2f}"


# ======================================================
# HERO HEADER
# ======================================================

st.markdown("""
<div class="hero-header">
    <p class="hero-title">📈 AI Stock Market Predictor</p>
    <p class="hero-subtitle">
        <span class="live-dot"></span>
        Indian & Global Market Analysis • Deep Learning LSTM • Technical Indicators • Live News
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.markdown('<p class="sidebar-header">⚡ Market Selection</p>', unsafe_allow_html=True)

    market = st.selectbox(
        "🌐 Choose Market",
        ["NSE", "BSE", "US", "INDEX"],
        help="Select the stock exchange"
    )

    if market == "INDEX":
        index = st.selectbox(
            "📊 Choose Index",
            ["NIFTY50", "BANKNIFTY", "SENSEX"]
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
            "Tata Motors": "TATAMOTORS",
            "Wipro": "WIPRO",
            "Bajaj Finance": "BAJFINANCE",
            "Maruti Suzuki": "MARUTI"
        }

        company = st.selectbox(
            "⭐ Popular Stocks",
            list(popular.keys())
        )

        symbol = st.text_input(
            "🔍 Or Enter Symbol",
            popular[company]
        )

        if market == "NSE":
            ticker = symbol.upper() + ".NS"
        elif market == "BSE":
            ticker = symbol.upper() + ".BO"
        else:
            ticker = symbol.upper()

    st.markdown("---")

    start_date = st.date_input(
        "📅 Start Date",
        pd.to_datetime("2015-01-01")
    )

    end_date = st.date_input(
        "📅 End Date",
        pd.to_datetime("today")
    )

    st.markdown("---")

    # Stock Comparison Tool
    st.markdown('<p class="sidebar-header">📊 Stock Comparison</p>', unsafe_allow_html=True)

    enable_compare = st.checkbox("Enable Comparison", value=False)

    if enable_compare:
        compare_symbol = st.text_input(
            "Compare With",
            "TCS" if market != "INDEX" else "^NSEI"
        )
        if market == "NSE":
            compare_ticker = compare_symbol.upper() + ".NS"
        elif market == "BSE":
            compare_ticker = compare_symbol.upper() + ".BO"
        elif market == "INDEX":
            compare_ticker = compare_symbol.upper()
        else:
            compare_ticker = compare_symbol.upper()

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: var(--text-muted); font-size: 0.75rem;">
            Built with ❤️ by <span class="footer-brand">Sarthak Uniyal</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# DOWNLOAD DATA
# ======================================================

with st.spinner("⚡ Fetching live market data..."):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

# Fix yfinance MultiIndex issue
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in data.columns:
        if isinstance(data[col], pd.DataFrame):
            data[col] = data[col].iloc[:, 0]

if data.empty:
    st.error("❌ No stock data found. Please check the symbol and try again.")
    st.stop()

# ======================================================
# STOCK METRICS
# ======================================================

latest_price = float(np.array(data["Close"]).flatten()[-1])
prev_price = float(np.array(data["Close"]).flatten()[-2]) if len(data) > 1 else latest_price
day_high = float(np.array(data["High"]).flatten()[-1])
day_low = float(np.array(data["Low"]).flatten()[-1])
day_open = float(np.array(data["Open"]).flatten()[-1])

price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100

vol_arr = np.array(data["Volume"]).flatten() if "Volume" in data.columns else None
volume = float(vol_arr[-1]) if vol_arr is not None else 0

# 52-week stats
if len(data) >= 252:
    high_52w = float(np.array(data["High"]).flatten()[-252:].max())
    low_52w = float(np.array(data["Low"]).flatten()[-252:].min())
else:
    high_52w = float(np.array(data["High"]).flatten().max())
    low_52w = float(np.array(data["Low"]).flatten().min())

# Display ticker info
display_name = ticker.replace(".NS", "").replace(".BO", "")
delta_class = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
delta_icon = "▲" if price_change >= 0 else "▼"

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
    <span style="font-size: 1.3rem; font-weight: 700; color: var(--text-primary);">{display_name}</span>
    <span class="section-badge">
        <span class="live-dot"></span> LIVE
    </span>
    <span style="color: var(--text-muted); font-size: 0.9rem;">{market} Market</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">₹{latest_price:,.2f}</div>
        <div class="{delta_class}">{delta_icon} {abs(price_change):.2f} ({abs(pct_change):.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Day Open</div>
        <div class="metric-value">₹{day_open:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Day High</div>
        <div class="metric-value">₹{day_high:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Day Low</div>
        <div class="metric-value">₹{day_low:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">52W High</div>
        <div class="metric-value">₹{high_52w:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Volume</div>
        <div class="metric-value">{format_large_number(volume) if volume > 0 else 'N/A'}</div>
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
    st.markdown("""
    <div class="section-header">
        📊 Price Chart <span class="section-badge">INTERACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    chart_type = st.radio(
        "Chart Type",
        ["Candlestick", "Line", "Area"],
        horizontal=True
    )

    # Build chart with volume subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=None
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="rgba(16, 185, 129, 0.8)",
                decreasing_fillcolor="rgba(239, 68, 68, 0.8)",
                name="Price"
            ),
            row=1, col=1
        )
    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                line=dict(color="#6C63FF", width=2),
                name="Close Price",
                fill=None
            ),
            row=1, col=1
        )
    else:  # Area
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                line=dict(color="#6C63FF", width=2),
                fill="tozeroy",
                fillcolor="rgba(108, 99, 255, 0.15)",
                name="Close Price"
            ),
            row=1, col=1
        )

    # Volume bars
    if "Volume" in data.columns:
        colors = ["#10b981" if c >= o else "#ef4444"
                  for c, o in zip(data["Close"], data["Open"])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                marker_color=colors,
                opacity=0.5,
                name="Volume",
                showlegend=False
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Inter", color="#94a3b8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis2=dict(showgrid=False),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            title="Price (₹)"
        ),
        yaxis2=dict(
            gridcolor="rgba(255,255,255,0.04)",
            title="Volume"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2: INDICATORS
# ======================================================

with tab2:
    indicator_tabs = st.tabs(["Moving Averages", "RSI", "MACD", "Bollinger Bands"])

    # Moving Averages
    with indicator_tabs[0]:
        st.markdown("""
        <div class="section-header">
            📉 Moving Averages <span class="section-badge">MA50 • MA100 • MA200</span>
        </div>
        """, unsafe_allow_html=True)

        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA100"] = data["Close"].rolling(100).mean()
        data["MA200"] = data["Close"].rolling(200).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            name="Close", line=dict(color="#e2e8f0", width=1.5)
        ))
        fig_ma.add_trace(go.Scatter(
            x=data.index, y=data["MA50"],
            name="MA50", line=dict(color="#6C63FF", width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=data.index, y=data["MA100"],
            name="MA100", line=dict(color="#a78bfa", width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=data.index, y=data["MA200"],
            name="MA200", line=dict(color="#c084fc", width=2, dash="dot")
        ))

        fig_ma.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    # RSI
    with indicator_tabs[1]:
        st.markdown("""
        <div class="section-header">
            📊 Relative Strength Index <span class="section-badge">RSI-14</span>
        </div>
        """, unsafe_allow_html=True)

        rsi = calculate_rsi(data)
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
        fig_rsi.add_trace(go.Scatter(
            x=data.index, y=rsi,
            line=dict(color="#a78bfa", width=2),
            name="RSI"
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(59,130,246,0.5)", annotation_text="Oversold (30)")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(108,99,255,0.05)", line_width=0)

        fig_rsi.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 100]),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)")
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    with indicator_tabs[2]:
        st.markdown("""
        <div class="section-header">
            📈 MACD <span class="section-badge">12-26-9</span>
        </div>
        """, unsafe_allow_html=True)

        macd_line, signal_line, histogram = calculate_macd(data)

        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])

        fig_macd.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            line=dict(color="#e2e8f0", width=1.5), name="Price"
        ), row=1, col=1)

        fig_macd.add_trace(go.Scatter(
            x=data.index, y=macd_line,
            line=dict(color="#6C63FF", width=2), name="MACD"
        ), row=2, col=1)

        fig_macd.add_trace(go.Scatter(
            x=data.index, y=signal_line,
            line=dict(color="#ef4444", width=1.5), name="Signal"
        ), row=2, col=1)

        hist_colors = ["#10b981" if h >= 0 else "#ef4444" for h in histogram]
        fig_macd.add_trace(go.Bar(
            x=data.index, y=histogram,
            marker_color=hist_colors, opacity=0.6, name="Histogram", showlegend=False
        ), row=2, col=1)

        fig_macd.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis2=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    with indicator_tabs[3]:
        st.markdown("""
        <div class="section-header">
            📉 Bollinger Bands <span class="section-badge">20-Day SMA ± 2σ</span>
        </div>
        """, unsafe_allow_html=True)

        upper, middle, lower = calculate_bollinger(data)

        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=data.index, y=upper, name="Upper Band",
            line=dict(color="rgba(239,68,68,0.5)", width=1)
        ))
        fig_bb.add_trace(go.Scatter(
            x=data.index, y=lower, name="Lower Band",
            line=dict(color="rgba(59,130,246,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(108,99,255,0.05)"
        ))
        fig_bb.add_trace(go.Scatter(
            x=data.index, y=middle, name="SMA 20",
            line=dict(color="#a78bfa", width=1.5, dash="dot")
        ))
        fig_bb.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Close",
            line=dict(color="#e2e8f0", width=1.5)
        ))

        fig_bb.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_bb, use_container_width=True)


# ======================================================
# TAB 3: AI PREDICTION
# ======================================================

with tab3:
    st.markdown("""
    <div class="section-header">
        🤖 AI Prediction Engine <span class="section-badge">LSTM DEEP LEARNING</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        if not TF_AVAILABLE:
            st.warning("⚠️ TensorFlow not installed. Install with: `pip install tensorflow`")
            st.stop()

        if len(data) < 120:
            st.warning("⚠️ Not enough historical data for prediction (need at least 120 days).")
            st.stop()

        model = load_model("model/stock_lstm_model.keras")

        close_data = data[["Close"]]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        x_test = []
        y_test = []

        for i in range(100, len(scaled_data)):
            x_test.append(scaled_data[i - 100:i])
            y_test.append(scaled_data[i])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        with st.spinner("🧠 Running LSTM model inference..."):
            predictions = model.predict(x_test, verbose=0)

        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test)

        # Actual vs Predicted chart
        dates = data.index[-len(actual_prices):]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=dates, y=actual_prices.flatten(),
            name="Actual Price",
            line=dict(color="#e2e8f0", width=2)
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates, y=predictions.flatten(),
            name="Predicted Price",
            line=dict(color="#6C63FF", width=2)
        ))

        fig_pred.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(family="Inter", color="#94a3b8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Price (₹)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Next-Day Prediction
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
                <div class="prediction-text">Next Day Prediction: ₹{next_day_price:,.2f}</div>
                <div class="prediction-sub">
                    {trend_text} • Expected Change: ₹{change_val:+,.2f} ({change_pct:+.2f}%)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-Step Forecast (7 & 30 days)
        st.markdown("""
        <div class="section-header">
            🔮 Multi-Step Forecast <span class="section-badge">7 & 30 DAY</span>
        </div>
        """, unsafe_allow_html=True)

        forecast_col1, forecast_col2 = st.columns(2)

        # 7-Day Forecast
        forecast_7 = []
        current_input = scaled_data[-100:].copy()
        for _ in range(7):
            inp = np.reshape(current_input, (1, 100, 1))
            pred = model.predict(inp, verbose=0)
            forecast_7.append(scaler.inverse_transform(pred)[0][0])
            current_input = np.append(current_input[1:], pred, axis=0)

        # 30-Day Forecast
        forecast_30 = []
        current_input = scaled_data[-100:].copy()
        for _ in range(30):
            inp = np.reshape(current_input, (1, 100, 1))
            pred = model.predict(inp, verbose=0)
            forecast_30.append(scaler.inverse_transform(pred)[0][0])
            current_input = np.append(current_input[1:], pred, axis=0)

        with forecast_col1:
            st.markdown("#### 📅 7-Day Forecast")
            last_date = data.index[-1]
            forecast_dates_7 = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='B')

            fig_f7 = go.Figure()
            # Historical tail
            fig_f7.add_trace(go.Scatter(
                x=data.index[-30:], y=data["Close"].iloc[-30:].values.flatten(),
                name="Historical", line=dict(color="#e2e8f0", width=1.5)
            ))
            fig_f7.add_trace(go.Scatter(
                x=forecast_dates_7, y=forecast_7,
                name="7-Day Forecast", line=dict(color="#10b981", width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=6)
            ))
            fig_f7.update_layout(
                height=350, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter", color="#94a3b8"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                showlegend=False
            )
            st.plotly_chart(fig_f7, use_container_width=True)

            f7_change = forecast_7[-1] - latest_price
            f7_pct = (f7_change / latest_price) * 100
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <span style="color: {'#10b981' if f7_change >= 0 else '#ef4444'}; font-weight: 700; font-size: 1.2rem;">
                    ₹{forecast_7[-1]:,.2f} ({'▲' if f7_change >= 0 else '▼'} {abs(f7_pct):.2f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

        with forecast_col2:
            st.markdown("#### 📅 30-Day Forecast")
            forecast_dates_30 = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='B')

            fig_f30 = go.Figure()
            fig_f30.add_trace(go.Scatter(
                x=data.index[-60:], y=data["Close"].iloc[-60:].values.flatten(),
                name="Historical", line=dict(color="#e2e8f0", width=1.5)
            ))
            fig_f30.add_trace(go.Scatter(
                x=forecast_dates_30, y=forecast_30,
                name="30-Day Forecast", line=dict(color="#c084fc", width=2.5, dash="dot"),
                mode="lines+markers", marker=dict(size=4)
            ))
            fig_f30.update_layout(
                height=350, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter", color="#94a3b8"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                showlegend=False
            )
            st.plotly_chart(fig_f30, use_container_width=True)

            f30_change = forecast_30[-1] - latest_price
            f30_pct = (f30_change / latest_price) * 100
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <span style="color: {'#10b981' if f30_change >= 0 else '#ef4444'}; font-weight: 700; font-size: 1.2rem;">
                    ₹{forecast_30[-1]:,.2f} ({'▲' if f30_change >= 0 else '▼'} {abs(f30_pct):.2f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown("""
        <div class="glass-card">
            <p style="color: var(--text-secondary);">
                ⚠️ <strong>Model not loaded.</strong> Train and place your LSTM model at
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
    st.markdown("""
    <div class="section-header">
        📰 Market News Feed <span class="section-badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    try:
        search_term = ticker.replace(".NS", "").replace(".BO", "")
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={search_term}+stock+market"
        )

        if feed.entries:
            for i, article in enumerate(feed.entries[:12]):
                published = getattr(article, 'published', '')
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">📄 {article.title}</div>
                    <div class="news-meta">{published}</div>
                </div>
                """, unsafe_allow_html=True)
                st.link_button(
                    f"Read Article →",
                    article.link,
                    use_container_width=False
                )
        else:
            st.info("No news articles found for this stock.")

    except Exception:
        st.warning("⚠️ Unable to fetch news. Check your internet connection.")


# ======================================================
# TAB 5: STOCK COMPARISON
# ======================================================

with tab5:
    st.markdown("""
    <div class="section-header">
        🔄 Stock Comparison <span class="section-badge">HEAD-TO-HEAD</span>
    </div>
    """, unsafe_allow_html=True)

    if enable_compare:
        with st.spinner("Fetching comparison data..."):
            compare_data = yf.download(
                compare_ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True
            )

        if isinstance(compare_data.columns, pd.MultiIndex):
            compare_data.columns = compare_data.columns.get_level_values(0)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in compare_data.columns:
                if isinstance(compare_data[col], pd.DataFrame):
                    compare_data[col] = compare_data[col].iloc[:, 0]

        if not compare_data.empty:
            # Normalized comparison chart
            norm_main = (data["Close"] / data["Close"].iloc[0]) * 100
            norm_compare = (compare_data["Close"] / compare_data["Close"].iloc[0]) * 100

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=data.index, y=norm_main.values.flatten(),
                name=display_name, line=dict(color="#6C63FF", width=2.5)
            ))
            compare_display = compare_ticker.replace(".NS", "").replace(".BO", "")
            fig_compare.add_trace(go.Scatter(
                x=compare_data.index, y=norm_compare.values.flatten(),
                name=compare_display, line=dict(color="#10b981", width=2.5)
            ))

            fig_compare.update_layout(
                height=500,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter", color="#94a3b8"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Normalized Price (Base=100)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Comparison table
            c_latest = float(np.array(compare_data["Close"]).flatten()[-1])
            c_prev = float(np.array(compare_data["Close"]).flatten()[-2]) if len(compare_data) > 1 else c_latest
            c_change = ((c_latest - c_prev) / c_prev) * 100
            c_high = float(np.array(compare_data["High"]).flatten()[-1])
            c_low = float(np.array(compare_data["Low"]).flatten()[-1])

            st.markdown(f"""
            <table class="compare-table">
                <tr>
                    <th>Metric</th>
                    <th>{display_name}</th>
                    <th>{compare_display}</th>
                </tr>
                <tr>
                    <td>Current Price</td>
                    <td>₹{latest_price:,.2f}</td>
                    <td>₹{c_latest:,.2f}</td>
                </tr>
                <tr>
                    <td>Day Change</td>
                    <td style="color: {'#10b981' if pct_change >= 0 else '#ef4444'}">{pct_change:+.2f}%</td>
                    <td style="color: {'#10b981' if c_change >= 0 else '#ef4444'}">{c_change:+.2f}%</td>
                </tr>
                <tr>
                    <td>Day High</td>
                    <td>₹{day_high:,.2f}</td>
                    <td>₹{c_high:,.2f}</td>
                </tr>
                <tr>
                    <td>Day Low</td>
                    <td>₹{day_low:,.2f}</td>
                    <td>₹{c_low:,.2f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.error("Could not fetch comparison data. Check the symbol.")
    else:
        st.info("💡 Enable **Stock Comparison** in the sidebar to compare two stocks head-to-head.")


# ======================================================
# TAB 6: RAW DATA
# ======================================================

with tab6:
    st.markdown("""
    <div class="section-header">
        📋 Historical Data <span class="section-badge">RAW</span>
    </div>
    """, unsafe_allow_html=True)

    data_display = data.copy()
    data_display.index = data_display.index.strftime('%Y-%m-%d')

    show_rows = st.slider("Rows to display", 10, min(500, len(data)), 50)
    st.dataframe(
        data_display.tail(show_rows),
        use_container_width=True,
        height=500
    )

    # Download button
    csv = data.to_csv()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{display_name}_stock_data.csv",
        mime="text/csv"
    )

# ======================================================
# FOOTER
# ======================================================

st.markdown("""
<div class="footer">
    <p class="footer-text">
        Built with ❤️ by <span class="footer-brand">Sarthak Uniyal</span> •
        Powered by <span class="footer-brand">LSTM Deep Learning</span> •
        Data from <span class="footer-brand">Yahoo Finance</span>
    </p>
    <p class="footer-text" style="margin-top: 0.5rem; font-size: 0.75rem;">
        ⚠️ This tool is for educational purposes only. Do not use for actual trading decisions.
    </p>
</div>
""", unsafe_allow_html=True)