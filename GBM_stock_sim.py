import matplotlib.pyplot as plt
import numpy as np

# Define parameters
mu, n, T, M, S0, sigma = 0.1, 100, 1, 5000, 100, 0.3

dt = T / n  # time step
St = np.exp((mu - sigma ** 2 / 2) * dt
            + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)  # stock price change with GBM formula
"""
GBM formula: dS_t = mu * S_t * dt + sigma * S_t * dW_t
where:
    dS_t = change in stock price
    mu = drift
    S_t = stock price at time t
    dt = time step
    sigma = volatility
    dW_t = Wienner process

Wiener process can be defined as a random variable with a normal distribution with mean 0 and variance dt.
"""

St = np.vstack([np.ones(M), St])  # add the initial price to the beginning of the array
St = S0 * St.cumprod(axis=0)  # calculate the stock price at time t

time = np.linspace(0, T, n + 1)
tt = np.full(shape=(M, n + 1), fill_value=time).T

plus = sum(
    1 for x in St[-1, :] if x > S0)  # number of simulations where the stock price is higher than the initial price
mean = np.mean([np.log(x / S0) / T for x in St[-1, :]])  # mean of the log returns
print("Number of simulations where the stock price is higher than the initial price: {0}".format(plus))
print("Mean of the log returns: {0}".format(mean))

# Plot the simulations
plt.plot(tt, St)
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(
        S0, mu, sigma)
)
plt.show()
