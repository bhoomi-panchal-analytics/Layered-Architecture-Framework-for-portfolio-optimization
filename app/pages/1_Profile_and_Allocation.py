import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Investor Allocation Overview")

weights = pd.read_csv("results/final_weights.csv", index_col=0)
returns = pd.read_csv("results/full_model_returns.csv", index_col=0)

# 1 Pie Chart
fig1 = px.pie(values=weights.iloc[:,0], names=weights.index,
              title="Capital Allocation")
st.plotly_chart(fig1)

# 2 Risk-Return Scatter
asset_returns = returns.mean()*252
asset_vol = returns.std()*np.sqrt(252)
df = pd.DataFrame({"Return":asset_returns,
                   "Volatility":asset_vol})

fig2 = px.scatter(df, x="Volatility", y="Return",
                  title="Risk-Return Map")
st.plotly_chart(fig2)

# 3 Rolling Return
rolling_return = (1+returns.iloc[:,0]).cumprod()
st.line_chart(rolling_return)

# 4 Rolling Volatility
rolling_vol = returns.iloc[:,0].rolling(63).std()*np.sqrt(252)
st.line_chart(rolling_vol)

# Add more sector and frontier charts as needed
