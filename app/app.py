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
# 1️⃣ INVESTOR PROFILE
# =====================================================

st.sidebar.header("Investor Profile")

capital = st.sidebar.number_input("Capital ($)", 1000, 10000000, 100000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 3.0)/100

start_date = st.sidebar.date_input("Start Date", date(2018,1,1))
end_date = st.sidebar.date_input("End Date", date.today())

assets_input = st.sidebar.text_input(
    "Asset Universe (comma separated)",
    "AAPL,MSFT,JPM,XOM,NVDA"
)

assets = [x.strip().upper() for x in assets_input.split(",") if x.strip()]

# =====================================================
# 2️⃣ DATA LOADING (SAFE)
# =====================================================

@st.cache_data
def load_data(assets, start, end):
    prices = yf.download(assets, start=start, end=end)["Close"]
    return prices

prices = load_data(assets, start_date, end_date)

if prices.empty or len(prices) < 50:
    st.error("Insufficient market data. Adjust inputs.")
    st.stop()

returns = prices.pct_change().dropna()

benchmark = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
benchmark = benchmark.pct_change().dropna()

returns, benchmark = returns.align(benchmark, join="inner", axis=0)

if len(returns) < 30:
    st.error("Not enough overlapping benchmark data.")
    st.stop()

# =====================================================
# 3️⃣ BASELINE EQUAL WEIGHT PORTFOLIO
# =====================================================

weights = np.ones(len(assets)) / len(assets)
portfolio = returns @ weights

# =====================================================
# 4️⃣ PERFORMANCE DIAGNOSTICS
# =====================================================

st.header("Pre-Optimization Diagnostics")

ann_return = portfolio.mean() * 252
ann_vol = portfolio.std() * np.sqrt(252)

sharpe = 0 if ann_vol == 0 else (ann_return - risk_free_rate)/ann_vol

col1,col2,col3 = st.columns(3)
col1.metric("Annual Return", f"{ann_return:.2%}")
col2.metric("Annual Volatility", f"{ann_vol:.2%}")
col3.metric("Sharpe Ratio", round(sharpe,3))

# =====================================================
# 5️⃣ CORRELATION MATRIX
# =====================================================

st.subheader("Correlation Matrix")

corr = returns.corr().fillna(0)

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Asset Correlation Heatmap"
)

st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================
# 6️⃣ PORTFOLIO VS BENCHMARK
# =====================================================

st.header("Portfolio vs Benchmark")

cum_port = (1 + portfolio).cumprod()
cum_bench = (1 + benchmark).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio"))
fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name="Benchmark"))
fig.update_layout(title="Cumulative Return Comparison")
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 7️⃣ ROLLING ANALYTICS
# =====================================================

st.header("Rolling Risk Diagnostics")

window = st.slider("Rolling Window (Months)", 3, 24, 12)
window_days = window * 21

rolling_vol = portfolio.rolling(window_days).std()*np.sqrt(252)
rolling_sharpe = (
    portfolio.rolling(window_days).mean() /
    portfolio.rolling(window_days).std()
)*np.sqrt(252)

st.line_chart(rolling_vol.dropna())
st.line_chart(rolling_sharpe.replace([np.inf,-np.inf],0).dropna())

# =====================================================
# 8️⃣ DRAWDOWN ANALYSIS
# =====================================================

st.header("Drawdown Analysis")

cum = (1+portfolio).cumprod()
drawdown = cum / cum.cummax() - 1

st.line_chart(drawdown)

max_dd = drawdown.min()
st.metric("Max Drawdown", f"{max_dd:.2%}")

# =====================================================
# 9️⃣ RISK DISTRIBUTION
# =====================================================

st.header("Risk Distribution")

fig_hist = px.histogram(portfolio, nbins=50, title="Return Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

skewness = skew(portfolio)
kurt_val = kurtosis(portfolio)

col4,col5 = st.columns(2)
col4.metric("Skewness", round(skewness,3))
col5.metric("Kurtosis", round(kurt_val,3))

var_method = st.selectbox("VaR Method", ["Historical","Parametric"])

if var_method == "Historical":
    var95 = np.percentile(portfolio,5)
else:
    var95 = portfolio.mean() - 1.65*portfolio.std()

cvar95 = portfolio[portfolio <= var95].mean()

col6,col7 = st.columns(2)
col6.metric("VaR 95%", f"{var95:.2%}")
col7.metric("CVaR 95%", f"{cvar95:.2%}")

# =====================================================
# 10️⃣ ACTIVE MANAGEMENT METRICS
# =====================================================

st.header("Active Management Metrics")

tracking_error = (portfolio - benchmark).std()*np.sqrt(252)

info_ratio = 0 if tracking_error == 0 else \
    (ann_return - benchmark.mean()*252)/tracking_error

beta = np.cov(portfolio, benchmark)[0,1] / np.var(benchmark)
alpha = ann_return - beta*(benchmark.mean()*252)

col8,col9,col10 = st.columns(3)
col8.metric("Information Ratio", round(info_ratio,3))
col9.metric("Beta", round(beta,3))
col10.metric("Alpha", f"{alpha:.2%}")

# =====================================================
# 11️⃣ OPTIMIZATION BUTTON
# =====================================================

st.markdown("---")

if st.button("Run Optimization"):

    optimized_return = ann_return
    optimized_vol = ann_vol
    optimized_sharpe = sharpe

    st.success("Optimization Completed (Baseline Equal Weight)")

    col11,col12,col13 = st.columns(3)
    col11.metric("Optimized Return", f"{optimized_return:.2%}")
    col12.metric("Optimized Volatility", f"{optimized_vol:.2%}")
    col13.metric("Optimized Sharpe", round(optimized_sharpe,3))

    st.bar_chart(pd.Series(weights, index=assets))
