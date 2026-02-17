import numpy as np
import pandas as pd
from numpy.linalg import norm, cond, eigvals
from sklearn.covariance import LedoitWolf

class CovarianceValidator:

    def __init__(self, returns, window=252):
        self.returns = returns
        self.window = window

    def rolling_covariances(self):
        sample_covs = []
        lw_covs = []

        for i in range(self.window, len(self.returns)):
            window_data = self.returns.iloc[i-self.window:i]

            sample_cov = window_data.cov()
            lw = LedoitWolf().fit(window_data.values)
            lw_cov = pd.DataFrame(lw.covariance_,
                                  index=sample_cov.index,
                                  columns=sample_cov.columns)

            sample_covs.append(sample_cov)
            lw_covs.append(lw_cov)

        return sample_covs, lw_covs

    def covariance_stability(self):
        sample_covs, lw_covs = self.rolling_covariances()

        sample_diffs = []
        lw_diffs = []

        for i in range(1, len(sample_covs)):
            sample_diffs.append(
                norm(sample_covs[i] - sample_covs[i-1], 'fro')
            )
            lw_diffs.append(
                norm(lw_covs[i] - lw_covs[i-1], 'fro')
            )

        return np.mean(sample_diffs), np.mean(lw_diffs)

    def condition_numbers(self):
        sample_covs, lw_covs = self.rolling_covariances()

        sample_cond = np.mean([cond(cov) for cov in sample_covs])
        lw_cond = np.mean([cond(cov) for cov in lw_covs])

        return sample_cond, lw_cond
