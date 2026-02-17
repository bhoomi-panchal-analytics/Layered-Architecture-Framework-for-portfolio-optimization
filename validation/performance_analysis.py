import numpy as np
import pandas as pd

class PerformanceAnalyzer:

    def sharpe_ratio(self, returns, rf=0):
        excess = returns - rf
        return np.sqrt(252) * np.mean(excess) / np.std(excess)

    def max_drawdown(self, returns):
        cumulative = (1 + pd.Series(returns)).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        return drawdown.min()

    def calmar_ratio(self, returns):
        annual_return = np.mean(returns) * 252
        max_dd = abs(self.max_drawdown(returns))
        return annual_return / max_dd

    def weight_stability(self, weights):
        """
        Measure weight volatility over time
        """
        weight_changes = np.diff(weights, axis=0)
        return np.mean(np.abs(weight_changes))
