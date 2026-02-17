import numpy as np
from scipy.optimize import minimize

class EntropyScenario:

    def __init__(self, prior_probs):
        self.prior = prior_probs

    def update(self, A, b):

        def objective(p):
            return np.sum(p * np.log(p / self.prior))

        constraints = [{
            'type': 'eq',
            'fun': lambda p: A @ p - b
        }]

        bounds = [(1e-8, 1) for _ in self.prior]

        result = minimize(objective,
                          self.prior,
                          bounds=bounds,
                          constraints=constraints)

        return result.x
