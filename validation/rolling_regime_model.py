import numpy as np
import pandas as pd
from models.dynamic_risk import DynamicRiskEngine
from models.bl_dynamic import BlackLittermanDynamic
from models.esg_optimizer import ESGOptimizer
from models.regime_detection import RegimeDetector

class RollingRegimeModel:

    def __init__(self, returns, esg_scores, window=252, rebalance=21):
        self.returns = returns
        self.esg = esg_scores
        self.window = window
        self.rebalance = rebalance

        self.regime_detector = RegimeDetector(returns)

    def run(self):

        regimes = self.regime_detector.classify_regime()
        portfolio_returns = []
        weights_history = []

        market_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        for i in range(self.window, len(self.returns), self.rebalance):

            train = self.returns.iloc[i-self.window:i]
            test = self.returns.iloc[i:i+self.rebalance]
            current_regime = regimes.iloc[i]

            mu_hist = train.mean().values

            risk_engine = DynamicRiskEngine(train)
            cov_dynamic = risk_engine.dynamic_covariance().values

            # Regime-dependent risk aversion
            if current_regime == "High Vol":
                risk_penalty = 1.0
            elif current_regime == "Low Vol":
                risk_penalty = 0.2
            else:
                risk_penalty = 0.5

            # Black-Litterman
            P = np.eye(len(mu_hist))
            Q = mu_hist

            bl = BlackLittermanDynamic(cov_dynamic, market_weights)
            mu_bl = bl.posterior(P, Q)

            # ESG Optimization
            optimizer = ESGOptimizer(mu_bl,
                                     risk_penalty * cov_dynamic,
                                     self.esg.values)

            w = optimizer.optimize(min_esg_score=60)
            weights_history.append(w)

            test_portfolio = test @ w
            portfolio_returns.extend(test_portfolio.values)

        return np.array(portfolio_returns), np.array(weights_history)
