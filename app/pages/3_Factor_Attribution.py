import streamlit as st
import pandas as pd

st.title("Factor Attribution")

factor_table = pd.read_csv("results/factor_attribution.csv",
                           index_col=0)

st.dataframe(factor_table)

st.bar_chart(factor_table)
