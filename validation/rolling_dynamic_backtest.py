import numpy as np
from models.dynamic_risk import DynamicRiskEngine

class RollingDynamicBacktester:

    def __init__(self, returns, window=252, rebalance=21):
        self.returns = returns
        self.window = window
        self.rebalance = rebalance

    def optimize_weights(self, mu, cov):
        inv_cov = np.linalg.inv(cov)
        w = inv_cov @ mu
        w = w / np.sum(w)
        return w

    def run(self):

        portfolio_returns = []
        weights_history = []

        for i in range(self.window, len(self.returns), self.rebalance):

            train = self.returns.iloc[i-self.window:i]
            test = self.returns.iloc[i:i+self.rebalance]

            mu = train.mean().values

            risk_engine = DynamicRiskEngine(train)
            dynamic_cov = risk_engine.dynamic_covariance().values

            w = self.optimize_weights(mu, dynamic_cov)
            weights_history.append(w)

            test_portfolio = test @ w
            portfolio_returns.extend(test_portfolio.values)

        return np.array(portfolio_returns), np.array(weights_history)
