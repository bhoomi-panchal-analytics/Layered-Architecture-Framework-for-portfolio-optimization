import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import skew, kurtosis, jarque_bera

st.title("Performance Intelligence Dashboard")

# ======================================================
# LOAD DATA
# ======================================================

assets = ["AAPL","MSFT","JPM","XOM","NVDA"]
prices = yf.download(assets, period="5y")["Close"]
returns = prices.pct_change().dropna()

benchmark = yf.download("^GSPC", period="5y")["Close"].pct_change().dropna()
benchmark = benchmark.loc[returns.index]

weights = np.ones(len(assets)) / len(assets)
portfolio = returns @ weights

rf = 0.03 / 252

# ======================================================
# TOP SECTION — PORTFOLIO VS BENCHMARK
# ======================================================

st.header("Portfolio vs Benchmark")

cum_port = (1 + portfolio).cumprod()
cum_bench = (1 + benchmark).cumprod()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio"))
fig1.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name="Benchmark"))
fig1.update_layout(title="Cumulative Return")
st.plotly_chart(fig1, use_container_width=True)

rolling_12m = portfolio.rolling(252).mean()*252
rolling_12m_b = benchmark.rolling(252).mean()*252

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=rolling_12m.index, y=rolling_12m, name="Portfolio"))
fig2.add_trace(go.Scatter(x=rolling_12m_b.index, y=rolling_12m_b, name="Benchmark"))
fig2.update_layout(title="Rolling 12M Return")
st.plotly_chart(fig2, use_container_width=True)

ann_return = portfolio.mean()*252
ann_vol = portfolio.std()*np.sqrt(252)
sharpe = (ann_return - 0.03)/ann_vol
sortino = (ann_return - 0.03)/(portfolio[portfolio<0].std()*np.sqrt(252))

tracking_error = (portfolio - benchmark).std()*np.sqrt(252)
info_ratio = (ann_return - benchmark.mean()*252)/tracking_error
beta = np.cov(portfolio, benchmark)[0,1]/np.var(benchmark)
alpha = ann_return - beta*(benchmark.mean()*252)

col1,col2,col3,col4 = st.columns(4)
col1.metric("Annual Return", round(ann_return,4))
col2.metric("Volatility", round(ann_vol,4))
col3.metric("Sharpe", round(sharpe,3))
col4.metric("Sortino", round(sortino,3))

col5,col6,col7,col8 = st.columns(4)
col5.metric("Information Ratio", round(info_ratio,3))
col6.metric("Tracking Error", round(tracking_error,4))
col7.metric("Beta", round(beta,3))
col8.metric("Alpha", round(alpha,4))

# ======================================================
# ROLLING RISK DIAGNOSTICS
# ======================================================

st.header("Rolling Risk Diagnostics")

window = st.slider("Rolling Window (Months)",3,24,12)
window_days = window*21

rolling_sharpe = (
    portfolio.rolling(window_days).mean() /
    portfolio.rolling(window_days).std()
)*np.sqrt(252)

rolling_vol = portfolio.rolling(window_days).std()*np.sqrt(252)

rolling_beta = portfolio.rolling(window_days).cov(benchmark) / benchmark.rolling(window_days).var()

rolling_corr = portfolio.rolling(window_days).corr(benchmark)

st.line_chart(rolling_sharpe)
st.line_chart(rolling_vol)
st.line_chart(rolling_beta)
st.line_chart(rolling_corr)

# ======================================================
# DRAWDOWN ANALYTICS
# ======================================================

st.header("Drawdown Analytics")

cum = (1+portfolio).cumprod()
drawdown = cum / cum.cummax() - 1

st.line_chart(drawdown)

max_dd = drawdown.min()

st.metric("Max Drawdown", round(max_dd,4))

# Worst 5 drawdowns
dd_table = drawdown.sort_values().head(5)
st.dataframe(dd_table)

# ======================================================
# RISK DISTRIBUTION
# ======================================================

st.header("Risk Distribution")

fig_hist = px.histogram(portfolio, nbins=50)
st.plotly_chart(fig_hist, use_container_width=True)

sk = skew(portfolio)
kt = kurtosis(portfolio)
jb_stat, jb_p = jarque_bera(portfolio)

var_method = st.selectbox("VaR Method",["Historical","Parametric"])

if var_method=="Historical":
    var95 = np.percentile(portfolio,5)
else:
    var95 = -(portfolio.mean() - 1.65*portfolio.std())

cvar95 = portfolio[portfolio<=var95].mean()

col9,col10,col11,col12 = st.columns(4)
col9.metric("Skewness",round(sk,3))
col10.metric("Kurtosis",round(kt,3))
col11.metric("Jarque-Bera",round(jb_stat,2))
col12.metric("VaR 95%",round(var95,4))

st.metric("CVaR 95%",round(cvar95,4))

# ======================================================
# ATTRIBUTION PREVIEW
# ======================================================

st.header("Attribution Preview")

contrib_return = returns.mean()*weights*252
st.bar_chart(contrib_return)

cov = returns.cov()*252
mcr = (cov @ weights) / np.sqrt(weights.T @ cov @ weights)
st.bar_chart(pd.Series(mcr,index=assets))

# ======================================================
# REGIME PERFORMANCE
# ======================================================

st.header("Regime Performance")

vix = yf.download("^VIX",period="5y")["Close"]
vix = vix.loc[portfolio.index]

high_vix = portfolio[vix>vix.median()]
low_vix = portfolio[vix<=vix.median()]

col13,col14 = st.columns(2)
col13.metric("Sharpe (High VIX)",
             round((high_vix.mean()*252)/(high_vix.std()*np.sqrt(252)),3))
col14.metric("Sharpe (Low VIX)",
             round((low_vix.mean()*252)/(low_vix.std()*np.sqrt(252)),3))
