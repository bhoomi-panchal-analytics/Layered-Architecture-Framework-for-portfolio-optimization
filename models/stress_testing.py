import numpy as np
import pandas as pd

class StressTesting:

    def __init__(self, returns, weights, cov_matrix):
        self.returns = returns
        self.weights = weights
        self.cov = cov_matrix

    def volatility_shock(self, multiplier=2.0):
        shocked_cov = self.cov * multiplier
        portfolio_vol = np.sqrt(self.weights.T @ shocked_cov @ self.weights)
        return portfolio_vol

    def correlation_shock(self, target_corr=0.8):

        std = np.sqrt(np.diag(self.cov))
        corr_matrix = self.cov / np.outer(std, std)

        shocked_corr = np.full_like(corr_matrix, target_corr)
        np.fill_diagonal(shocked_corr, 1)

        shocked_cov = np.outer(std, std) * shocked_corr

        portfolio_vol = np.sqrt(self.weights.T @ shocked_cov @ self.weights)
        return portfolio_vol

    def carbon_shock(self, carbon_intensity, penalty=0.05):
        """
        Penalize high carbon assets
        """
        adjusted_returns = self.returns.copy()

        for i, asset in enumerate(self.returns.columns):
            adjusted_returns[asset] -= penalty * carbon_intensity[i]

        stressed_portfolio = adjusted_returns @ self.weights
        return stressed_portfolio.mean() * 252
