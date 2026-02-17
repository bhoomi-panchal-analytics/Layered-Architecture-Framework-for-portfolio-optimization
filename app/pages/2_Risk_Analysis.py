import streamlit as st
import pandas as pd

st.title("Risk Decomposition")

risk_table = pd.read_csv("results/risk_decomposition.csv",
                         index_col=0)

st.dataframe(risk_table)

st.bar_chart(risk_table["Percent_Risk_Contribution"])
