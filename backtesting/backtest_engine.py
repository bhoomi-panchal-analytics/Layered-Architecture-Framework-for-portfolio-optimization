import pandas as pd
import numpy as np

class BacktestEngine:

    def __init__(self, returns):
        self.returns = returns

    def run_backtest(self, weights):
        portfolio_returns = self.returns @ weights
        cumulative = (1 + portfolio_returns).cumprod()
        return cumulative

    def sharpe_ratio(self, portfolio_returns, rf=0):
        excess = portfolio_returns - rf
        return np.sqrt(252) * excess.mean() / excess.std()

    def calmar_ratio(self, portfolio_returns):
        cumulative = (1 + portfolio_returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        max_dd = drawdown.min()
        return portfolio_returns.mean() * 252 / abs(max_dd)
