import streamlit as st
import pandas as pd
from experiments.experiment_runner import ExperimentRunner

st.title("Hyperparameter Sensitivity Lab")

tau = st.slider("Black-Litterman Tau", 0.01, 0.3, 0.05)
esg_threshold = st.slider("Minimum ESG Score", 40, 80, 60)

if st.button("Run Experiment"):

    runner = ExperimentRunner()
    results = runner.run()

    st.write(results)
