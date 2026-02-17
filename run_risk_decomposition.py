import pandas as pd
from models.risk_decomposition import RiskDecomposition

returns = pd.read_csv("data/processed/returns.csv", index_col=0)
cov = returns.cov().values

weights = pd.read_csv("results/final_weights.csv").values.flatten()

rd = RiskDecomposition(weights, cov)
risk_table = rd.summary(returns.columns)

risk_table.to_csv("results/risk_decomposition.csv")
print(risk_table)
