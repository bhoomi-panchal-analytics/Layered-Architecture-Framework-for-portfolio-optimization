import cvxpy as cp
import numpy as np

class Optimizer:

    def __init__(self, expected_returns, cov):
        self.mu = expected_returns.values
        self.cov = cov.values
        self.n = len(self.mu)

    def max_sharpe(self, rf=0.0):
        w = cp.Variable(self.n)

        objective = cp.Maximize(
            (self.mu - rf) @ w
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return w.value
