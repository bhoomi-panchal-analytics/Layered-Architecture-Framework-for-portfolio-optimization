import numpy as np
import pandas as pd
from reporting.performance_report import PerformanceReport

class BootstrapValidator:

    def __init__(self, returns, weights, n_bootstrap=500):
        self.returns = returns
        self.weights = weights
        self.n_bootstrap = n_bootstrap

    def block_bootstrap(self, block_size=21):

        portfolio_returns = []

        for _ in range(self.n_bootstrap):

            indices = []
            while len(indices) < len(self.returns):
                start = np.random.randint(0, len(self.returns)-block_size)
                indices.extend(range(start, start+block_size))

            indices = indices[:len(self.returns)]
            sampled = self.returns.iloc[indices]

            port = sampled @ self.weights
            portfolio_returns.append(port.mean()*252)

        return np.array(portfolio_returns)
