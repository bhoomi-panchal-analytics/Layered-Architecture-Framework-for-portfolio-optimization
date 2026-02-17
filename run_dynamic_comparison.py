import pandas as pd
from validation.rolling_backtest import RollingBacktester
from validation.rolling_dynamic_backtest import RollingDynamicBacktester
from validation.performance_analysis import PerformanceAnalyzer

returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)

# Static LW
static_bt = RollingBacktester(returns)
static_returns, static_weights = static_bt.run(use_ledoit=True)

# Dynamic
dynamic_bt = RollingDynamicBacktester(returns)
dynamic_returns, dynamic_weights = dynamic_bt.run()

analyzer = PerformanceAnalyzer()

results = {
    "Static_Sharpe": analyzer.sharpe_ratio(static_returns),
    "Dynamic_Sharpe": analyzer.sharpe_ratio(dynamic_returns),
    "Static_Calmar": analyzer.calmar_ratio(static_returns),
    "Dynamic_Calmar": analyzer.calmar_ratio(dynamic_returns),
    "Static_Weight_Instability": analyzer.weight_stability(static_weights),
    "Dynamic_Weight_Instability": analyzer.weight_stability(dynamic_weights)
}

print(results)

pd.Series(results).to_csv("results/dynamic_vs_static.csv")
