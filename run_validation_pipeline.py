from validation.covariance_validation import CovarianceValidator
from validation.rolling_backtest import RollingBacktester
from validation.performance_analysis import PerformanceAnalyzer
import pandas as pd

# Load processed returns
returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)

# 1. Covariance Stability
cov_validator = CovarianceValidator(returns)
condition_df = cov_validator.compare_condition_numbers()
condition_df.to_csv("results/covariance_condition_numbers.csv")

# 2. Rolling Backtest
backtester = RollingBacktester(returns)
sample_returns, sample_weights = backtester.run(use_ledoit=False)
lw_returns, lw_weights = backtester.run(use_ledoit=True)

# 3. Performance Comparison
analyzer = PerformanceAnalyzer()

results = {
    "Sample_Sharpe": analyzer.sharpe_ratio(sample_returns),
    "LW_Sharpe": analyzer.sharpe_ratio(lw_returns),
    "Sample_Calmar": analyzer.calmar_ratio(sample_returns),
    "LW_Calmar": analyzer.calmar_ratio(lw_returns),
    "Sample_Weight_Instability": analyzer.weight_stability(sample_weights),
    "LW_Weight_Instability": analyzer.weight_stability(lw_weights)
}

pd.Series(results).to_csv("results/performance_comparison.csv")

print(results)
