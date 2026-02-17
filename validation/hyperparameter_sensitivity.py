import numpy as np
import pandas as pd
from validation.rolling_full_model import RollingFullModel
from reporting.performance_report import PerformanceReport

class HyperparameterSensitivity:

    def __init__(self, returns, esg_scores):
        self.returns = returns
        self.esg = esg_scores

    def test_esg_threshold(self, thresholds=[50,60,70]):

        results = {}

        for t in thresholds:
            model = RollingFullModel(self.returns, self.esg)
            portfolio_returns, _ = model.run()

            report = PerformanceReport(portfolio_returns)
            results[f"ESG_{t}"] = report.sharpe()

        return pd.Series(results)
