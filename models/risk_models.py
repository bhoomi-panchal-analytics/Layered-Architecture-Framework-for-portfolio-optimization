import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from arch import arch_model

class RiskEngine:

    def __init__(self, returns):
        self.returns = returns

    def historical_covariance(self):
        return self.returns.cov()

    def ledoit_wolf_covariance(self):
        lw = LedoitWolf()
        lw.fit(self.returns.values)
        return pd.DataFrame(lw.covariance_,
                            index=self.returns.columns,
                            columns=self.returns.columns)

    def garch_volatility(self, asset):
        """
        Univariate GARCH(1,1) forecast
        """
        am = arch_model(self.returns[asset]*100, vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1)
        variance = forecast.variance.iloc[-1, 0]
        return np.sqrt(variance) / 100
