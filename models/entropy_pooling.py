import numpy as np

class EntropyPooling:

    def __init__(self, prior_probs):
        self.prior = prior_probs

    def update_probabilities(self, constraints, A):
        """
        Solve:
        min sum(p log(p/prior))
        s.t. A p = constraints
        """
        from scipy.optimize import minimize

        def objective(p):
            return np.sum(p * np.log(p / self.prior))

        cons = ({
            'type': 'eq',
            'fun': lambda p: A @ p - constraints
        })

        bounds = [(1e-8, 1) for _ in self.prior]

        result = minimize(objective,
                          self.prior,
                          bounds=bounds,
                          constraints=cons)

        return result.x
