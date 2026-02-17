import numpy as np
import pandas as pd

class RegimeDetector:

    def __init__(self, returns, window=63):
        self.returns = returns
        self.window = window

    def compute_rolling_vol(self):
        return self.returns.rolling(self.window).std() * np.sqrt(252)

    def classify_regime(self):

        rolling_vol = self.compute_rolling_vol().mean(axis=1)

        high_threshold = rolling_vol.quantile(0.75)
        low_threshold = rolling_vol.quantile(0.25)

        regimes = pd.Series(index=rolling_vol.index)

        regimes[rolling_vol <= low_threshold] = "Low Vol"
        regimes[(rolling_vol > low_threshold) & (rolling_vol < high_threshold)] = "Medium Vol"
        regimes[rolling_vol >= high_threshold] = "High Vol"

        return regimes
