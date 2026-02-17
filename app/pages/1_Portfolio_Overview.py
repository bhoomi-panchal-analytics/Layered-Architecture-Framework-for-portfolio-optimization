import streamlit as st
import pandas as pd
from reporting.performance_report import PerformanceReport

st.title("Portfolio Performance Overview")

returns = pd.read_csv("results/full_model_returns.csv",
                      index_col=0,
                      parse_dates=True).iloc[:,0]

report = PerformanceReport(returns)
metrics = report.summary()

col1, col2, col3 = st.columns(3)

col1.metric("Sharpe Ratio", round(metrics["Sharpe"], 3))
col2.metric("Sortino Ratio", round(metrics["Sortino"], 3))
col3.metric("Calmar Ratio", round(metrics["Calmar"], 3))

st.line_chart((1+returns).cumprod())
