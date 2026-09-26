# 📈 AI-Powered Stock Market Prediction & Analysis Platform

An AI-driven stock market prediction web application built using Deep Learning (LSTM), Streamlit, and Yahoo Finance APIs. The platform enables users to analyze Indian and global stocks, visualize historical trends, calculate technical indicators, and forecast future stock prices using Machine Learning.

---

## 🚀 Features

### Market Support

* NSE Stocks
* BSE Stocks
* NIFTY50
* BANKNIFTY
* SENSEX
* US Stocks

### AI Prediction

* Deep Learning LSTM Model
* Next-Day Stock Price Prediction
* 7-Day Multi-Step Forecast
* 30-Day Multi-Step Forecast
* Historical Trend Analysis
* Data Normalization & Forecasting

### Technical Indicators

* Moving Average (MA50 / MA100 / MA200)
* Relative Strength Index (RSI-14)
* MACD (12-26-9) with Signal Line & Histogram
* Bollinger Bands (20-Day SMA ± 2σ)

### Visualization

* Interactive Candlestick / Line / Area Charts
* Volume Overlay with Color-Coded Bars
* Prediction vs Actual Graphs
* Technical Indicator Charts
* Real-Time Dashboard with Live Metrics
* Stock Comparison Tool (Normalized)

### News Analytics

* Real-Time Financial News Feed
* Stock-Specific Market Updates

### Premium UI

* Glassmorphism Dark Theme
* Gradient Accents & Animations
* Color-Coded Price Deltas
* Tabbed Layout with Sub-Tabs
* Responsive Design

---

## 🛠️ Tech Stack

### Frontend

* Streamlit (with Custom CSS)

### Backend

* Python

### Machine Learning

* TensorFlow
* Keras
* LSTM Neural Networks
* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Data Sources

* Yahoo Finance API

### Visualization

* Plotly

---

## 📂 Project Structure

```
Stock-Market-Prediction/
├── app.py                          # Main application
├── model/
│   └── stock_lstm_model.keras      # Trained LSTM model
├── assets/
│   └── logo.png                    # Logo asset
├── src/
│   ├── charts.py                   # Chart utilities
│   ├── data_loader.py              # Data fetching
│   ├── indicators.py               # Technical indicators
│   ├── news.py                     # News feed
│   └── predictor.py                # Prediction logic
├── pages/
│   ├── 1_Dashboard.py              # Dashboard page
│   ├── 2_Predictions.py            # Predictions page
│   └── 3_News.py                   # News page
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── .python-version                 # Python version for deployment
├── requirements.txt                # Dependencies
├── train_model.py                  # Model training script
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Sarthak149-uni/stock-market-prediction.git
cd stock-market-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model (Optional)

```bash
python train_model.py
```

### Run Application

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (including the `model/` directory)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository: `Sarthak149-uni/stock-market-prediction`
5. Set Main file path: `app.py`
6. Click **Deploy**

---

## 📈 Future Enhancements

* Portfolio Tracker
* Watchlist Management
* Sentiment Analysis using FinBERT
* Stock Recommendation Engine
* User Authentication

---

## 👨‍💻 Author

Sarthak Uniyal

LinkedIn: https://www.linkedin.com/in/sarthakuniyalus

GitHub: https://github.com/Sarthak149-uni
