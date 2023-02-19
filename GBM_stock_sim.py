import numpy as np
import matplotlib.pyplot as plt

mu, n, T, M, S0, sigma=0.1, 100, 1, 5000, 100, 0.3

dt= T/n
St = np.exp((mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
St = np.vstack([np.ones(M), St])
St = S0 * St.cumprod(axis=0)

time = np.linspace(0,T,n+1)
tt = np.full(shape=(M,n+1), fill_value=time).T

plus=sum(1 for x in St[-1,:] if x>S0)
mean=np.mean([np.log(x/S0)/T for x in St[-1,:]])
print(plus, mean.round(3))

plt.plot(tt, St)
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma)
)
plt.show()
