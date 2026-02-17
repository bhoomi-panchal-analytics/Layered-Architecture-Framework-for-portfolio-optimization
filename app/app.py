import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from scipy.stats import skew, kurtosis

st.set_page_config(layout="wide")
st.title("Quantitative Portfolio Optimization Platform")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Investor Settings")

capital = st.sidebar.number_input("Capital ($)", 1000, 10000000, 100000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100

start_date = st.sidebar.date_input("Start Date", date(2018,1,1))
end_date = st.sidebar.date_input("End Date", date.today())

assets_input = st.sidebar.text_input(
    "Asset Universe (comma separated)",
    "AAPL,MSFT,JPM,XOM,NVDA"
)

assets = [x.strip().upper() for x in assets_input.split(",") if x.strip()]

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_prices(assets, start, end):
    data = yf.download(assets, start=start, end=end)["Close"]
    return data

prices = load_prices(assets, start_date, end_date)

if prices.empty or len(prices) < 50:
    st.error("Not enough market data. Adjust date range or assets.")
    st.stop()

returns = prices.pct_change().dropna()

benchmark = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
benchmark = benchmark.pct_change().dropna()

# Align safely
returns, benchmark = returns.align(benchmark, join="inner", axis=0)

if len(returns) < 30:
    st.error("Not enough overlapping benchmark data.")
    st.stop()

# =====================================================
# EQUAL WEIGHT PORTFOLIO
# =====================================================

weights = np.ones(len(assets)) / len(assets)
portfolio = returns.dot(weights)

# =====================================================
# PERFORMANCE METRICS
# =====================================================

st.header("Performance Overview")

ann_return = float(portfolio.mean() * 252)
ann_vol = float(portfolio.std() * np.sqrt(252))

sharpe = 0.0
if ann_vol > 1e-8:
    sharpe = (ann_return - risk_free_rate) / ann_vol

col1, col2, col3 = st.columns(3)
col1.metric("Annual Return", f"{ann_return:.2%}")
col2.metric("Annual Volatility", f"{ann_vol:.2%}")
col3.metric("Sharpe Ratio", round(sharpe,3))

# =====================================================
# CORRELATION MATRIX
# =====================================================

st.subheader("Correlation Matrix")

corr = returns.corr().fillna(0)

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Asset Correlation"
)

st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================
# PORTFOLIO VS BENCHMARK
# =====================================================

st.header("Portfolio vs Benchmark")

cum_port = (1 + portfolio).cumprod()
cum_bench = (1 + benchmark).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio"))
fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name="Benchmark"))
fig.update_layout(title="Cumulative Performance")
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ACTIVE METRICS
# =====================================================

tracking_error = float((portfolio - benchmark).std() * np.sqrt(252))

if tracking_error > 1e-8:
    info_ratio = (ann_return - float(benchmark.mean()*252)) / tracking_error
else:
    info_ratio = 0.0

beta = float(np.cov(portfolio, benchmark)[0,1] / np.var(benchmark))
alpha = ann_return - beta * float(benchmark.mean()*252)

col4, col5, col6 = st.columns(3)
col4.metric("Information Ratio", round(info_ratio,3))
col5.metric("Beta", round(beta,3))
col6.metric("Alpha", f"{alpha:.2%}")

# =====================================================
# ROLLING ANALYTICS
# =====================================================

st.header("Rolling Diagnostics")

window = st.slider("Rolling Window (Months)", 3, 24, 12)
window_days = window * 21

rolling_vol = portfolio.rolling(window_days).std() * np.sqrt(252)
rolling_sharpe = (
    portfolio.rolling(window_days).mean() /
    portfolio.rolling(window_days).std()
) * np.sqrt(252)

st.line_chart(rolling_vol.dropna())
st.line_chart(rolling_sharpe.replace([np.inf, -np.inf], np.nan).dropna())

# =====================================================
# DRAWDOWN
# =====================================================

st.header("Drawdown")

cum = (1 + portfolio).cumprod()
drawdown = cum / cum.cummax() - 1

st.line_chart(drawdown)

max_dd = float(drawdown.min())
st.metric("Maximum Drawdown", f"{max_dd:.2%}")

# =====================================================
# RISK DISTRIBUTION
# =====================================================

st.header("Risk Distribution")

fig_hist = px.histogram(portfolio, nbins=50)
st.plotly_chart(fig_hist, use_container_width=True)

skewness = float(skew(portfolio))
kurt_val = float(kurtosis(portfolio))

col7, col8 = st.columns(2)
col7.metric("Skewness", round(skewness,3))
col8.metric("Kurtosis", round(kurt_val,3))

var_method = st.selectbox("VaR Method", ["Historical","Parametric"])

if var_method == "Historical":
    var95 = float(np.percentile(portfolio,5))
else:
    var95 = float(portfolio.mean() - 1.65*portfolio.std())

cvar95 = float(portfolio[portfolio <= var95].mean())

col9, col10 = st.columns(2)
col9.metric("VaR 95%", f"{var95:.2%}")
col10.metric("CVaR 95%", f"{cvar95:.2%}")

# =====================================================
# OPTIMIZATION BUTTON
# =====================================================

st.markdown("---")

if st.button("Run Optimization"):
    st.success("Baseline Equal-Weight Optimization Completed")
    st.bar_chart(pd.Series(weights, index=assets))
