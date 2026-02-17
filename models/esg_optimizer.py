import cvxpy as cp
import numpy as np

class ESGOptimizer:

    def __init__(self, expected_returns, cov_matrix, esg_scores):
        self.mu = expected_returns
        self.cov = cov_matrix
        self.esg = esg_scores
        self.n = len(self.mu)

    def optimize(self, min_esg_score=60):

        w = cp.Variable(self.n)

        portfolio_return = self.mu @ w
        portfolio_risk = cp.quad_form(w, self.cov)

        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            self.esg @ w >= min_esg_score
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value
