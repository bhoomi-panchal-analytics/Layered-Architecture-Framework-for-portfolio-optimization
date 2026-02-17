import pandas as pd
from utils.config_loader import load_config
from utils.seed_control import set_global_seed
from validation.rolling_regime_model import RollingRegimeModel
from reporting.performance_report import PerformanceReport
from models.transaction_cost import TransactionCostModel

class ExperimentRunner:

    def __init__(self, config_path="config/base_config.yaml"):
        self.config = load_config(config_path)
        set_global_seed(self.config["random_seed"])

    def run(self):

        returns = pd.read_csv("data/processed/returns.csv",
                              index_col=0,
                              parse_dates=True)

        esg = pd.read_csv("data/raw/esg_scores.csv",
                          index_col=0)

        model = RollingRegimeModel(
            returns,
            esg,
            window=self.config["data"]["window"],
            rebalance=self.config["data"]["rebalance"]
        )

        portfolio_returns, weights = model.run()

        # Apply transaction cost
        tc_model = TransactionCostModel()
        adjusted_returns = tc_model.apply_costs(
            portfolio_returns,
            weights,
            self.config["transaction_cost"]["cost_per_turnover"]
        )

        report = PerformanceReport(adjusted_returns)
        summary = report.summary()

        self.log_results(summary)

        return summary

    def log_results(self, results):

        log_path = "results/experiment_logs.csv"

        df = pd.DataFrame([results])

        try:
            existing = pd.read_csv(log_path)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass

        df.to_csv(log_path, index=False)
