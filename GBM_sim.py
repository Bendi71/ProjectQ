import numpy as np

def GBM_sim(S0, r, sig, T, N):
    dt = T / N
    t = np.linspace(0, T, N)
    S = S0 * np.exp((r - 0.5 * sig ** 2) * t + sig * np.cumsum(np.random.standard_normal(size=N)) * np.sqrt(dt))
    return S