import numpy as np
import pandas as pd
from models.dynamic_risk import DynamicRiskEngine
from models.bl_dynamic import BlackLittermanDynamic
from models.esg_optimizer import ESGOptimizer

class RollingFullModel:

    def __init__(self, returns, esg_scores, window=252, rebalance=21):
        self.returns = returns
        self.esg = esg_scores
        self.window = window
        self.rebalance = rebalance

    def run(self):

        portfolio_returns = []
        weights_history = []

        market_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        for i in range(self.window, len(self.returns), self.rebalance):

            train = self.returns.iloc[i-self.window:i]
            test = self.returns.iloc[i:i+self.rebalance]

            mu_hist = train.mean().values

            risk_engine = DynamicRiskEngine(train)
            cov_dynamic = risk_engine.dynamic_covariance().values

            # Black-Litterman
            P = np.eye(len(mu_hist))  # identity views example
            Q = mu_hist               # historical view proxy

            bl = BlackLittermanDynamic(cov_dynamic, market_weights)
            mu_bl = bl.posterior(P, Q)

            # ESG optimization
            optimizer = ESGOptimizer(mu_bl,
                                     cov_dynamic,
                                     self.esg.values)

            w = optimizer.optimize(min_esg_score=60)
            weights_history.append(w)

            test_portfolio = test @ w
            portfolio_returns.extend(test_portfolio.values)

        return np.array(portfolio_returns), np.array(weights_history)
