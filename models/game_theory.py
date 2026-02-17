import numpy as np

class GameTheoryAllocator:

    def __init__(self, returns, cov):
        self.returns = returns
        self.cov = cov

    def nash_equilibrium(self, lambda_risk=0.5):
        """
        Risk player vs Return player
        """
        inv_cov = np.linalg.inv(self.cov)
        w = inv_cov @ self.returns
        w = w / np.sum(w)

        # risk penalty blending
        w = (1-lambda_risk)*w + lambda_risk*(np.ones(len(w))/len(w))
        return w
