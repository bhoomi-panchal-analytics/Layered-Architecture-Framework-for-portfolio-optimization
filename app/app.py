import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Quantitative Portfolio Optimization Platform")

# =====================================================
# 1️⃣ INVESTOR PROFILE & OBJECTIVE FUNCTION
# =====================================================

st.header("Investor Profile & Objective")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Investor Name")
    age = st.number_input("Age", 18, 100, 30)
    capital = st.number_input("Investment Capital ($)", 1000, 10000000, 100000)

with col2:
    objective = st.selectbox(
        "Objective Function",
        [
            "Maximize Sharpe Ratio",
            "Minimize Volatility",
            "Maximize Return",
            "Risk Parity",
            "Target Return Optimization",
            "Black-Litterman",
            "CVaR Minimization"
        ]
    )

    expected_return_model = st.selectbox(
        "Expected Return Model",
        [
            "Historical Mean",
            "CAPM",
            "Black-Litterman",
            "ARIMA Forecast"
        ]
    )

with col3:
    covariance_method = st.selectbox(
        "Covariance Estimation",
        [
            "Sample Covariance",
            "Ledoit-Wolf Shrinkage",
            "EWMA",
            "GARCH-based"
        ]
    )

    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100

# Save to session
st.session_state["capital"] = capital
st.session_state["objective"] = objective

# =====================================================
# 2️⃣ CONSTRAINT ENGINE
# =====================================================

st.header("Constraints & Risk Preferences")

col4, col5, col6 = st.columns(3)

with col4:
    risk_tolerance = st.slider("Risk Tolerance", 1, 10, 5)
    min_weight = st.number_input("Minimum Weight per Asset", 0.0, 1.0, 0.0)
    max_weight = st.number_input("Maximum Weight per Asset", 0.0, 1.0, 0.4)

with col5:
    leverage_allowed = st.toggle("Allow Leverage?")
    short_allowed = st.toggle("Allow Short Selling?")
    transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 5.0, 0.1) / 100

with col6:
    turnover_penalty = st.slider("Turnover Penalty", 0.0, 1.0, 0.1)
    liquidity_filter = st.toggle("Apply Liquidity Filter?")
    sector_constraint = st.toggle("Apply Sector Constraints?")

# =====================================================
# 3️⃣ MARKET DATA CONTROLS
# =====================================================

st.header("Market Data Controls")

col7, col8, col9 = st.columns(3)

with col7:
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

with col8:
    frequency = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly"])
    benchmark_choice = st.selectbox(
        "Benchmark",
        ["S&P 500", "NIFTY 50", "Custom"]
    )

with col9:
    regime_filter = st.selectbox(
        "Volatility Regime",
        ["All", "High VIX", "Low VIX"]
    )
    esg_filter = st.toggle("Apply ESG Filter?")

# =====================================================
# LOAD MARKET DATA
# =====================================================

assets = ["AAPL", "MSFT", "JPM", "XOM", "NVDA"]

prices = yf.download(
    assets,
    start=start_date,
    end=end_date
)["Close"]

returns = prices.pct_change().dropna()

# =====================================================
# 4️⃣ DIAGNOSTIC METRICS BEFORE OPTIMIZATION
# =====================================================

st.header("Pre-Optimization Diagnostics")

annual_return = returns.mean() * 252
annual_vol = returns.std() * np.sqrt(252)
sharpe = (annual_return.mean() - risk_free_rate) / annual_vol.mean()

col10, col11, col12 = st.columns(3)

col10.metric("Mean Annual Return", round(annual_return.mean(), 4))
col11.metric("Annualized Volatility", round(annual_vol.mean(), 4))
col12.metric("Sharpe Ratio (Naive)", round(sharpe, 3))

# Correlation heatmap
corr = returns.corr()
fig_corr = px.imshow(corr, title="Correlation Matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# Drawdown
portfolio_equal = returns.mean(axis=1)
cum = (1 + portfolio_equal).cumprod()
drawdown = cum / cum.cummax() - 1

st.line_chart(drawdown)

# VaR & CVaR
var_95 = np.percentile(portfolio_equal, 5)
cvar_95 = portfolio_equal[portfolio_equal <= var_95].mean()

col13, col14 = st.columns(2)
col13.metric("VaR (95%)", round(var_95, 4))
col14.metric("CVaR (95%)", round(cvar_95, 4))

# =====================================================
# 5️⃣ ADVANCED QUANT TOGGLES
# =====================================================

st.header("Advanced Quant Controls")

col15, col16, col17 = st.columns(3)

with col15:
    entropy_toggle = st.toggle("Entropy Regularization?")
    shrinkage_intensity = st.slider("Shrinkage Intensity", 0.0, 1.0, 0.2)

with col16:
    rolling_stability = st.toggle("Rolling Stability Analysis?")
    bootstrap_toggle = st.toggle("Bootstrap Validation?")

with col17:
    rebalance_frequency = st.selectbox(
        "Rebalancing Frequency",
        ["Monthly", "Quarterly", "Annual"]
    )

# =====================================================
# 6️⃣ STRESS TEST PANEL
# =====================================================

st.header("Stress Testing")

col18, col19, col20 = st.columns(3)

with col18:
    shock_returns = st.slider("Return Shock (%)", -20, 20, 0)

with col19:
    volatility_spike = st.slider("Volatility Spike (%)", 0, 100, 0)

with col20:
    crisis_toggle = st.selectbox(
        "Crisis Simulation",
        ["None", "2008 Crisis", "2020 COVID"]
    )

# =====================================================
# RUN OPTIMIZATION BUTTON
# =====================================================

st.markdown("---")
if st.button("Run Optimization"):

    st.success("Optimization Running Based on Selected Parameters")

    # Placeholder optimization logic
    weights = np.ones(len(assets)) / len(assets)

    optimized_return = np.dot(weights, annual_return)
    optimized_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    optimized_sharpe = (optimized_return - risk_free_rate) / optimized_vol

    st.metric("Optimized Return", round(optimized_return, 4))
    st.metric("Optimized Volatility", round(optimized_vol, 4))
    st.metric("Optimized Sharpe", round(optimized_sharpe, 3))

    st.bar_chart(pd.Series(weights, index=assets))
