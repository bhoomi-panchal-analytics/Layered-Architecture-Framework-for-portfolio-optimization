import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from pathlib import Path

st.title("Strategic Portfolio Allocation")

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_ASSETS = ["AAPL", "MSFT", "JPM", "XOM", "NVDA"]
CAPITAL = st.session_state.get("capital", 100000)

# -----------------------------
# LOAD RETURNS (OR FETCH LIVE)
# -----------------------------
@st.cache_data
def load_returns():
    try:
        returns = pd.read_csv(
            "data/processed/returns.csv",
            index_col=0,
            parse_dates=True
        )
    except:
        prices = yf.download(DEFAULT_ASSETS, period="5y")["Close"]
        returns = prices.pct_change().dropna()
    return returns

returns = load_returns()
assets = returns.columns.tolist()

# -----------------------------
# LOAD WEIGHTS (SAFE FALLBACK)
# -----------------------------
weights_path = Path("results/final_weights.csv")

if weights_path.exists():
    weights = pd.read_csv(weights_path, index_col=0).iloc[:,0]
else:
    st.warning("Final weights not found. Using equal-weight allocation.")
    weights = pd.Series(
        np.ones(len(assets)) / len(assets),
        index=assets
    )

# Normalize
weights = weights / weights.sum()

# -----------------------------
# CAPITAL DEPLOYMENT
# -----------------------------
allocation = weights * CAPITAL

allocation_df = pd.DataFrame({
    "Weight": weights,
    "Capital ($)": allocation
})

st.subheader("Capital Allocation")
st.dataframe(allocation_df)

# -----------------------------
# PIE CHART
# -----------------------------
fig_pie = px.pie(
    values=allocation,
    names=weights.index,
    title="Capital Distribution"
)
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# RISK-RETURN MAP
# -----------------------------
annual_return = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)

rr_df = pd.DataFrame({
    "Annual Return": annual_return,
    "Annual Volatility": annual_vol
})

fig_rr = px.scatter(
    rr_df,
    x="Annual Volatility",
    y="Annual Return",
    text=rr_df.index,
    title="Risk-Return Map"
)

st.plotly_chart(fig_rr, use_container_width=True)

# -----------------------------
# PORTFOLIO GROWTH
# -----------------------------
portfolio_returns = (returns @ weights).dropna()
growth = (1 + portfolio_returns).cumprod() * CAPITAL

st.subheader("Portfolio Growth")
st.line_chart(growth)

# -----------------------------
# ROLLING VOLATILITY
# -----------------------------
rolling_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)

st.subheader("Rolling Volatility (3M)")
st.line_chart(rolling_vol)

# -----------------------------
# VALUE AT RISK
# -----------------------------
var_95 = np.percentile(portfolio_returns, 5)
es_95 = portfolio_returns[portfolio_returns <= var_95].mean()

col1, col2 = st.columns(2)
col1.metric("Portfolio VaR (95%)", round(var_95, 4))
col2.metric("Expected Shortfall (95%)", round(es_95, 4))

# -----------------------------
# ASSET INDICATORS TABLE
# -----------------------------
benchmark = yf.download("^GSPC", period="5y")["Close"].pct_change().dropna()

betas = []
for asset in assets:
    aligned = pd.concat([returns[asset], benchmark], axis=1).dropna()
    beta = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1] / np.var(aligned.iloc[:,1])
    betas.append(beta)

indicator_df = pd.DataFrame({
    "Annual Return": annual_return,
    "Annual Volatility": annual_vol,
    "Beta vs S&P500": betas
})

st.subheader("Asset Risk Indicators")
st.dataframe(indicator_df)

# -----------------------------
# SHARPE
# -----------------------------
sharpe = (portfolio_returns.mean() /
          portfolio_returns.std()) * np.sqrt(252)

st.metric("Portfolio Sharpe Ratio", round(sharpe, 3))
