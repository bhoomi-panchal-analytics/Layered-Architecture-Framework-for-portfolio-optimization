def sharpe_ratio(returns, rf=0):
    excess = returns - rf
    return np.sqrt(252) * excess.mean() / excess.std()
