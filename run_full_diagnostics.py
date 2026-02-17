import pandas as pd
from validation.model_diagnostics import ModelDiagnostics

# Load stored results
condition_df = pd.read_csv("results/covariance_condition_numbers.csv")

sample_returns = pd.read_csv("results/sample_returns.csv").values.flatten()
lw_returns = pd.read_csv("results/lw_returns.csv").values.flatten()

sample_weights = pd.read_csv("results/sample_weights.csv").values
lw_weights = pd.read_csv("results/lw_weights.csv").values

diagnostics = ModelDiagnostics(
    condition_df,
    sample_returns,
    lw_returns,
    sample_weights,
    lw_weights
)

condition_results = diagnostics.check_condition_numbers()
sharpe_results = diagnostics.check_sharpe_improvement()
weight_results = diagnostics.check_weight_stability()
stability_results = diagnostics.overall_stability_assessment()

print("Condition Check:", condition_results)
print("Sharpe Check:", sharpe_results)
print("Weight Stability:", weight_results)
print("Overall Stability:", stability_results)
