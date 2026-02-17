import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.covariance import LedoitWolf

class DynamicRiskEngine:

    def __init__(self, returns):
        self.returns = returns

    # -----------------------------
    # 1. UNIVARIATE GARCH VOL
    # -----------------------------
    def forecast_garch_vol(self, asset):

        series = self.returns[asset] * 100
        model = arch_model(series, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")

        forecast = res.forecast(horizon=1)
        variance = forecast.variance.iloc[-1, 0]

        return np.sqrt(variance) / 100

    # -----------------------------
    # 2. FORECAST ALL VOLS
    # -----------------------------
    def forecast_all_vols(self):

        vols = {}
        for asset in self.returns.columns:
            vols[asset] = self.forecast_garch_vol(asset)

        return pd.Series(vols)

    # -----------------------------
    # 3. CORRELATION ESTIMATION
    # -----------------------------
    def correlation_matrix(self):

        lw = LedoitWolf().fit(self.returns.values)
        cov = lw.covariance_

        std_dev = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_dev, std_dev)

        return pd.DataFrame(corr,
                            index=self.returns.columns,
                            columns=self.returns.columns)

    # -----------------------------
    # 4. DYNAMIC COVARIANCE MATRIX
    # -----------------------------
    def dynamic_covariance(self):

        vols = self.forecast_all_vols()
        corr = self.correlation_matrix()

        D = np.diag(vols.values)
        dynamic_cov = D @ corr.values @ D

        return pd.DataFrame(dynamic_cov,
                            index=self.returns.columns,
                            columns=self.returns.columns)
