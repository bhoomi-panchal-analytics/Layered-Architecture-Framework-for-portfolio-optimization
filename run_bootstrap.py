import pandas as pd
from validation.bootstrap_validation import BootstrapValidator

returns = pd.read_csv("data/processed/returns.csv", index_col=0)
weights = pd.read_csv("results/final_weights.csv").values.flatten()

bootstrap = BootstrapValidator(returns, weights)
distribution = bootstrap.block_bootstrap()

print("Bootstrap Mean Return:", distribution.mean())
print("Bootstrap 5% Quantile:", np.percentile(distribution, 5))
