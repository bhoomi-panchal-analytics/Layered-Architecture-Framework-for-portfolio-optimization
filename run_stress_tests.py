import pandas as pd
from models.stress_testing import StressTesting

returns = pd.read_csv("data/processed/returns.csv", index_col=0)
weights = pd.read_csv("results/final_weights.csv").values.flatten()
cov = returns.cov().values
carbon_intensity = pd.read_csv("data/raw/esg_scores.csv")["Environmental"].values / 100

stress = StressTesting(returns, weights, cov)

vol_shock = stress.volatility_shock()
corr_shock = stress.correlation_shock()
carbon_shock_return = stress.carbon_shock(carbon_intensity)

print({
    "Volatility_Shock_Portfolio_Vol": vol_shock,
    "Correlation_Shock_Portfolio_Vol": corr_shock,
    "Carbon_Shock_Annual_Return": carbon_shock_return
})
