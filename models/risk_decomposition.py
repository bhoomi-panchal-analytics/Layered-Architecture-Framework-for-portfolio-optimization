import numpy as np
import pandas as pd

class RiskDecomposition:

    def __init__(self, weights, cov_matrix):
        self.w = np.array(weights)
        self.cov = np.array(cov_matrix)

    def portfolio_volatility(self):
        return np.sqrt(self.w.T @ self.cov @ self.w)

    def marginal_contribution(self):
        sigma_p = self.portfolio_volatility()
        mcr = (self.cov @ self.w) / sigma_p
        return mcr

    def component_contribution(self):
        mcr = self.marginal_contribution()
        return self.w * mcr

    def percentage_contribution(self):
        ccr = self.component_contribution()
        sigma_p = self.portfolio_volatility()
        return ccr / sigma_p

    def summary(self, asset_names):
        return pd.DataFrame({
            "Weight": self.w,
            "MCR": self.marginal_contribution(),
            "Component_Risk": self.component_contribution(),
            "Percent_Risk_Contribution": self.percentage_contribution()
        }, index=asset_names)
