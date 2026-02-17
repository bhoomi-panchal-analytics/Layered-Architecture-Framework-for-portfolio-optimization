import streamlit as st
import pandas as pd

st.title("ESG Exposure")

esg = pd.read_csv("data/raw/esg_scores.csv",
                  index_col=0)

st.dataframe(esg)

st.bar_chart(esg["Composite_ESG"])
