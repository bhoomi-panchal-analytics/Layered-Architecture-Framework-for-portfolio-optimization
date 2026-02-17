import streamlit as st
import pandas as pd

st.title("Stress Testing Results")

stress_results = pd.read_csv("results/stress_results.csv",
                             index_col=0)

st.dataframe(stress_results)
