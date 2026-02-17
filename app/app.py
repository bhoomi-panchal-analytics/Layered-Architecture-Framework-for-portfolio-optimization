import streamlit as st

st.set_page_config(page_title="Quant Portfolio Intelligence Lab",
                   layout="wide")

st.title("Quantitative Portfolio Intelligence Platform")

st.sidebar.header("User Profile")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 18, 100)
capital = st.sidebar.number_input("Capital ($)", 1000, 10000000, 100000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (1=Low,10=High)", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.write(f"Investor: {name}")
st.sidebar.write(f"Capital: ${capital:,.0f}")
st.sidebar.write(f"Risk Tolerance: {risk_tolerance}")

st.markdown("""
This platform integrates:
- Dynamic GARCH covariance
- Black–Litterman allocation
- ESG constraints
- Entropy scenario adjustment
- Regime switching
- Stress testing
- Bootstrap validation
""")
