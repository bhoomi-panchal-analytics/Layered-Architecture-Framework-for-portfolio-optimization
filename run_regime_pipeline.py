import pandas as pd
from validation.rolling_regime_model import RollingRegimeModel
from reporting.performance_report import PerformanceReport

returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
esg = pd.read_csv("data/raw/esg_scores.csv", index_col=0)

model = RollingRegimeModel(returns, esg)
portfolio_returns, weights = model.run()

report = PerformanceReport(portfolio_returns)
results = report.summary()

print(results)
pd.Series(results).to_csv("results/regime_model_performance.csv")
