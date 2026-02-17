import numpy as np
import pandas as pd

class BlackLitterman:

    def __init__(self, cov, market_weights, delta=2.5):
        self.cov = cov
        self.market_weights = market_weights
        self.delta = delta

    def implied_equilibrium_returns(self):
        return self.delta * self.cov @ self.market_weights

    def posterior_returns(self, P, Q, tau=0.05, omega=None):
        pi = self.implied_equilibrium_returns()

        if omega is None:
            omega = np.diag(np.diag(P @ (tau*self.cov) @ P.T))

        middle = np.linalg.inv(
            np.linalg.inv(tau*self.cov) + P.T @ np.linalg.inv(omega) @ P
        )

        posterior = middle @ (
            np.linalg.inv(tau*self.cov) @ pi +
            P.T @ np.linalg.inv(omega) @ Q
        )

        return posterior
