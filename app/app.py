import streamlit as st

st.set_page_config(page_title="Quant Portfolio Lab",
                   layout="wide")

st.title("Integrated Quantitative Portfolio Optimization Framework")

st.markdown("""
This dashboard integrates:

- Dynamic GARCH covariance
- Black–Litterman Bayesian allocation
- ESG constraints
- Regime switching
- Stress testing
- Bootstrap robustness
""")

st.sidebar.success("Select a page from the sidebar.")
