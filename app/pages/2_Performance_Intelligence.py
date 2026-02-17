import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

st.title("Performance Intelligence Dashboard")

# Load portfolio returns
portfolio = pd.read_csv("results/full_model_returns.csv",
                        index_col=0,
                        parse_dates=True).iloc[:,0]

portfolio.name = "Portfolio"

# Benchmark (S&P 500)
benchmark = yf.download("^GSPC",
                        start=portfolio.index.min())["Close"].pct_change()
benchmark = benchmark.loc[portfolio.index]
benchmark.name = "Benchmark"

# ------------------------------------------------
# 1. CUMULATIVE RETURN COMPARISON
# ------------------------------------------------
cum_port = (1 + portfolio).cumprod()
cum_bench = (1 + benchmark).cumprod()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cum_port.index,
                          y=cum_port,
                          name="Portfolio"))
fig1.add_trace(go.Scatter(x=cum_bench.index,
                          y=cum_bench,
                          name="Benchmark"))
fig1.update_layout(title="Cumulative Return vs Benchmark")
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# 2. LOG SCALE GROWTH
# ------------------------------------------------
fig2 = px.line(np.log(cum_port), title="Log-Scale Growth")
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# 3. ROLLING 12-MONTH RETURN
# ------------------------------------------------
rolling_12m = portfolio.rolling(252).mean()*252
fig3 = px.line(rolling_12m,
               title="Rolling 12-Month Annualized Return")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# 4. ROLLING SHARPE
# ------------------------------------------------
rolling_sharpe = (
    portfolio.rolling(252).mean() /
    portfolio.rolling(252).std()
) * np.sqrt(252)

fig4 = px.line(rolling_sharpe,
               title="Rolling 12-Month Sharpe Ratio")
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# 5. ROLLING SORTINO
# ------------------------------------------------
downside = portfolio.copy()
downside[downside > 0] = 0

rolling_sortino = (
    portfolio.rolling(252).mean() /
    downside.rolling(252).std()
) * np.sqrt(252)

fig5 = px.line(rolling_sortino,
               title="Rolling 12-Month Sortino Ratio")
st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------
# 6. DRAWDOWN CURVE
# ------------------------------------------------
cum = (1 + portfolio).cumprod()
drawdown = cum / cum.cummax() - 1

fig6 = px.line(drawdown,
               title="Drawdown Curve")
st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------
# 7. MONTHLY RETURN HEATMAP
# ------------------------------------------------
monthly = portfolio.resample("M").sum()
monthly_df = monthly.to_frame(name="Return")
monthly_df["Year"] = monthly_df.index.year
monthly_df["Month"] = monthly_df.index.month

pivot = monthly_df.pivot("Year", "Month", "Return")

fig7 = px.imshow(pivot,
                 title="Monthly Return Heatmap",
                 aspect="auto")
st.plotly_chart(fig7, use_container_width=True)

# ------------------------------------------------
# 8. RETURN DISTRIBUTION
# ------------------------------------------------
fig8 = px.histogram(portfolio,
                    nbins=50,
                    title="Return Distribution")
st.plotly_chart(fig8, use_container_width=True)

# ------------------------------------------------
# 9. VAR & EXPECTED SHORTFALL
# ------------------------------------------------
var_95 = np.percentile(portfolio, 5)
es_95 = portfolio[portfolio <= var_95].mean()

col1, col2 = st.columns(2)
col1.metric("VaR (95%)", round(var_95,4))
col2.metric("Expected Shortfall (95%)", round(es_95,4))

# ------------------------------------------------
# 10. ALPHA vs BETA SCATTER
# ------------------------------------------------
beta = np.cov(portfolio, benchmark)[0,1] / np.var(benchmark)
alpha = portfolio.mean()*252 - beta*(benchmark.mean()*252)

st.metric("Beta vs Benchmark", round(beta,3))
st.metric("Annual Alpha", round(alpha,4))
