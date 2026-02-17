import numpy as np

def sharpe_difference_test(r1, r2):

    diff = np.mean(r1) - np.mean(r2)
    var1 = np.var(r1)
    var2 = np.var(r2)

    n = len(r1)

    z = diff / np.sqrt((var1 + var2) / n)

    return z
