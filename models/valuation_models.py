import numpy as np

class ValuationEngine:

    def __init__(self, financials):
        self.financials = financials

    def dcf_model(self, fcf, growth, wacc, terminal_growth, years=5):
        projected_fcf = []
        for t in range(1, years+1):
            projected_fcf.append(fcf * ((1 + growth) ** t))

        terminal_value = (
            projected_fcf[-1] * (1 + terminal_growth)
        ) / (wacc - terminal_growth)

        discount_factors = [(1 / (1 + wacc) ** t) for t in range(1, years+1)]

        pv_fcf = sum([projected_fcf[i] * discount_factors[i] for i in range(years)])
        pv_terminal = terminal_value / ((1 + wacc) ** years)

        return pv_fcf + pv_terminal
