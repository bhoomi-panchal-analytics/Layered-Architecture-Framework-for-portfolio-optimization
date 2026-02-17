import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.title("Profile & Strategic Allocation Dashboard")

# ------------------------------
# LOAD DATA
# ------------------------------
weights = pd.read_csv("results/final_weights.csv", index_col=0)
returns = pd.read_csv("data/processed/returns.csv",
                      index_col=0,
                      parse_dates=True)

portfolio_returns = pd.read_csv("results/full_model_returns.csv",
                                index_col=0,
                                parse_dates=True).iloc[:,0]

esg = pd.read_csv("data/raw/esg_scores.csv",
                  index_col=0)

assets = weights.index.tolist()

# ------------------------------
# USER CAPITAL ALLOCATION
# ------------------------------
st.subheader("Capital Deployment")

capital = st.session_state.get("capital", 100000)

allocation_values = weights.iloc[:,0] * capital

allocation_df = pd.DataFrame({
    "Weight": weights.iloc[:,0],
    "Capital Allocation ($)": allocation_values
})

st.dataframe(allocation_df)

# ------------------------------
# PIE CHART ALLOCATION
# ------------------------------
fig1 = px.pie(values=allocation_values,
              names=weights.index,
              title="Capital Allocation Breakdown")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# RISK-RETURN MAP
# ------------------------------
annual_return = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)

rr_df = pd.DataFrame({
    "Return": annual_return,
    "Volatility": annual_vol
})

fig2 = px.scatter(rr_df,
                  x="Volatility",
                  y="Return",
                  text=rr_df.index,
                  title="Risk-Return Map")

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# EFFICIENT FRONTIER SIMULATION
# ------------------------------
st.subheader("Efficient Frontier Simulation")

n_sim = 3000
sim_returns = []
sim_vol = []

cov = returns.cov()

for _ in range(n_sim):
    w = np.random.random(len(assets))
    w /= np.sum(w)

    r = np.dot(w, annual_return)
    v = np.sqrt(np.dot(w.T, np.dot(cov*252, w)))

    sim_returns.append(r)
    sim_vol.append(v)

frontier_df = pd.DataFrame({
    "Return": sim_returns,
    "Volatility": sim_vol
})

fig3 = px.scatter(frontier_df,
                  x="Volatility",
                  y="Return",
                  title="Monte Carlo Efficient Frontier",
                  opacity=0.3)

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# PORTFOLIO VALUE OVER TIME
# ------------------------------
st.subheader("Portfolio Growth")

portfolio_growth = (1 + portfolio_returns).cumprod() * capital
st.line_chart(portfolio_growth)

# ------------------------------
# ROLLING VOLATILITY
# ------------------------------
rolling_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)
st.line_chart(rolling_vol)

# ------------------------------
# VAR PER ASSET
# ------------------------------
st.subheader("Asset Risk Indicators")

var_95 = returns.quantile(0.05)

beta_list = []
for asset in assets:
    benchmark = yf.download("^GSPC",
                             start=returns.index.min())["Close"].pct_change()
    aligned = pd.concat([returns[asset], benchmark], axis=1).dropna()
    beta = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1] / np.var(aligned.iloc[:,1])
    beta_list.append(beta)

indicator_df = pd.DataFrame({
    "Annual Return": annual_return,
    "Annual Volatility": annual_vol,
    "VaR (95%)": var_95,
    "Beta vs S&P500": beta_list,
    "ESG Score": esg["Composite_ESG"]
})

st.dataframe(indicator_df)

# ------------------------------
# SECTOR EXPOSURE (MANUAL MAP)
# ------------------------------
sector_map = {
    "AAPL":"Tech",
    "MSFT":"Tech",
    "JPM":"Finance",
    "XOM":"Energy",
    "NVDA":"Tech"
}

sector_series = pd.Series({
    asset: sector_map.get(asset, "Other")
    for asset in assets
})

sector_weights = allocation_df.groupby(sector_series)["Capital Allocation ($)"].sum()

fig4 = px.bar(sector_weights,
              title="Sector Exposure")
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# PORTFOLIO SHARPE
# ------------------------------
sharpe = (portfolio_returns.mean() /
          portfolio_returns.std()) * np.sqrt(252)

st.metric("Portfolio Sharpe Ratio", round(sharpe, 3))
