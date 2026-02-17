import numpy as np

class TransactionCostModel:

    def turnover(self, weights):
        weight_changes = np.diff(weights, axis=0)
        return np.sum(np.abs(weight_changes), axis=1)

    def apply_costs(self, returns, weights, cost_per_turnover=0.001):

        turnover_series = self.turnover(weights)
        adjusted_returns = returns.copy()

        for i in range(1, len(turnover_series)):
            adjusted_returns[i] -= cost_per_turnover * turnover_series[i-1]

        return adjusted_returns
