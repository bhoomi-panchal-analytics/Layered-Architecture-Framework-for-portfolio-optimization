import pandas as pd
from models.factor_attribution import FactorAttribution

portfolio_returns = pd.read_csv("results/full_model_returns.csv",
                                index_col=0,
                                parse_dates=True).iloc[:,0]

factors = pd.read_csv("data/raw/ff_factors.csv",
                      index_col=0,
                      parse_dates=True)

fa = FactorAttribution(portfolio_returns, factors)

contributions, r2 = fa.contribution_breakdown()

contributions.to_csv("results/factor_attribution.csv")

print("Factor Contributions:")
print(contributions)
print("R-squared:", r2)
