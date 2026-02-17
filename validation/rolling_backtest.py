import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class RollingBacktest:

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
        results = {
            "equal": [],
            "sample": [],
            "lw": []
        }

        for i in range(self.window, len(self.returns)-self.rebalance, self.rebalance):
            train = self.returns.iloc[i-self.window:i]
            test = self.returns.iloc[i:i+self.rebalance]

            mu = train.mean().values

            sample_cov = train.cov().values
            lw_cov = LedoitWolf().fit(train.values).covariance_

            w_equal = np.ones(len(mu)) / len(mu)
            w_sample = self.optimize_weights(mu, sample_cov)
            w_lw = self.optimize_weights(mu, lw_cov)

            results["equal"].extend(test.values @ w_equal)
            results["sample"].extend(test.values @ w_sample)
            results["lw"].extend(test.values @ w_lw)

        return pd.DataFrame(results)
