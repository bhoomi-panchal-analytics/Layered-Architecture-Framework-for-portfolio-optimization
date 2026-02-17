import numpy as np
import pandas as pd

class BlackLittermanDynamic:

    def __init__(self, cov_matrix, market_weights, delta=2.5):
        self.cov = cov_matrix
        self.market_weights = market_weights
        self.delta = delta

    def implied_returns(self):
        return self.delta * self.cov @ self.market_weights

    def posterior(self, P, Q, tau=0.05):

        pi = self.implied_returns()
        omega = np.diag(np.diag(P @ (tau*self.cov) @ P.T))

        middle = np.linalg.inv(
            np.linalg.inv(tau*self.cov) + P.T @ np.linalg.inv(omega) @ P
        )

        mu_bl = middle @ (
            np.linalg.inv(tau*self.cov) @ pi +
            P.T @ np.linalg.inv(omega) @ Q
        )

        return mu_bl
