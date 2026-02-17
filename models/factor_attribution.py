import pandas as pd
import statsmodels.api as sm

class FactorAttribution:

    def __init__(self, portfolio_returns, factor_data):
        self.portfolio_returns = portfolio_returns
        self.factor_data = factor_data

    def run_regression(self):

        aligned = self.factor_data.loc[self.portfolio_returns.index]
        X = sm.add_constant(aligned)
        y = self.portfolio_returns

        model = sm.OLS(y, X).fit()

        return model

    def contribution_breakdown(self):

        model = self.run_regression()
        params = model.params

        factor_means = self.factor_data.mean()

        contributions = {}

        contributions["Alpha"] = params["const"]

        for factor in factor_means.index:
            contributions[factor] = params[factor] * factor_means[factor]

        return pd.Series(contributions), model.rsquared
