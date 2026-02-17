import numpy as np
import pandas as pd

class PerformanceReport:

    def __init__(self, returns):
        self.returns = pd.Series(returns)

    def sharpe(self):
        return np.sqrt(252) * self.returns.mean() / self.returns.std()

    def sortino(self):
        downside = self.returns[self.returns < 0]
        return np.sqrt(252) * self.returns.mean() / downside.std()

    def max_drawdown(self):
        cumulative = (1 + self.returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        return drawdown.min()

    def calmar(self):
        annual_return = self.returns.mean() * 252
        return annual_return / abs(self.max_drawdown())

    def summary(self):
        return {
            "Sharpe": self.sharpe(),
            "Sortino": self.sortino(),
            "Calmar": self.calmar(),
            "Max_Drawdown": self.max_drawdown(),
            "Annual_Return": self.returns.mean() * 252,
            "Annual_Volatility": self.returns.std() * np.sqrt(252)
        }
