import numpy as np
import pandas as pd

class ModelDiagnostics:

    def __init__(self, condition_df,
                 sample_returns, lw_returns,
                 sample_weights, lw_weights):

        self.condition_df = condition_df
        self.sample_returns = sample_returns
        self.lw_returns = lw_returns
        self.sample_weights = sample_weights
        self.lw_weights = lw_weights

    # -------------------------------
    # 1. CONDITION NUMBER CHECK
    # -------------------------------
    def check_condition_numbers(self):

        sample_mean = self.condition_df["Sample_Condition"].mean()
        lw_mean = self.condition_df["LedoitWolf_Condition"].mean()

        improvement = (sample_mean - lw_mean) / sample_mean

        result = {
            "Sample_Avg_Condition": sample_mean,
            "LW_Avg_Condition": lw_mean,
            "Condition_Improvement_%": improvement * 100,
            "LW_Is_More_Stable": lw_mean < sample_mean
        }

        return result

    # -------------------------------
    # 2. SHARPE RATIO COMPARISON
    # -------------------------------
    def sharpe_ratio(self, returns):
        return np.sqrt(252) * np.mean(returns) / np.std(returns)

    def check_sharpe_improvement(self, threshold=0.15):

        sample_sharpe = self.sharpe_ratio(self.sample_returns)
        lw_sharpe = self.sharpe_ratio(self.lw_returns)

        improvement = lw_sharpe - sample_sharpe

        result = {
            "Sample_Sharpe": sample_sharpe,
            "LW_Sharpe": lw_sharpe,
            "Sharpe_Difference": improvement,
            "Improvement_Above_Threshold": improvement > threshold
        }

        return result

    # -------------------------------
    # 3. WEIGHT INSTABILITY CHECK
    # -------------------------------
    def weight_instability(self, weights):
        weight_changes = np.diff(weights, axis=0)
        return np.mean(np.abs(weight_changes))

    def check_weight_stability(self):

        sample_instability = self.weight_instability(self.sample_weights)
        lw_instability = self.weight_instability(self.lw_weights)

        result = {
            "Sample_Instability": sample_instability,
            "LW_Instability": lw_instability,
            "LW_More_Stable": lw_instability < sample_instability
        }

        return result

    # -------------------------------
    # 4. STABILITY WARNING SYSTEM
    # -------------------------------
    def overall_stability_assessment(self):

        sharpe_info = self.check_sharpe_improvement()
        weight_info = self.check_weight_stability()

        warning_flag = False

        if sharpe_info["Sharpe_Difference"] > 0 and \
           weight_info["LW_Instability"] > 1.5 * weight_info["Sample_Instability"]:

            warning_flag = True

        result = {
            "Sharpe_Improved": sharpe_info["Sharpe_Difference"] > 0,
            "Weight_Explosion_Risk": warning_flag,
            "Model_Is_Stable":
                sharpe_info["Sharpe_Difference"] > 0 and
                weight_info["LW_More_Stable"]
        }

        return result
