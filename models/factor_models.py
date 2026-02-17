import statsmodels.api as sm
import pandas as pd

class FactorModel:

    def __init__(self, returns, factors):
        self.returns = returns
        self.factors = factors

    def fama_french_expected_returns(self):
        expected_returns = {}

        for asset in self.returns.columns:
            y = self.returns[asset].dropna()
            X = self.factors.loc[y.index]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()
            betas = model.params

            factor_means = self.factors.mean()
            er = betas[0] + (betas[1:] * factor_means).sum()

            expected_returns[asset] = er

        return pd.Series(expected_returns)
