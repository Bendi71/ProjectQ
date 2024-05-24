"""
Geometric Brownian Motion Stock Price Simulation Project

This project aims to simulate stock prices using the Geometric Brownian Motion (GBM) model. GBM is a common model used in finance for assets that are assumed to have returns that are normally distributed. It is often used for stock price simulation because it can capture two important characteristics of stock price returns:

1. Drift: Over the long term, stock prices tend to increase. This is captured in the GBM model by the 'mu' parameter, which represents the expected return of the stock.

2. Volatility: Stock prices also exhibit variability or noise, which is captured in the GBM model by the 'sigma' parameter, representing the standard deviation of the stock's returns.

The GBM model assumes that the logarithmic returns of stock prices are normally distributed and that the prices themselves are log-normally distributed. This is a reasonable assumption for many assets and makes the model mathematically tractable.

In this project, we simulate the GBM using the formula:

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

where:
    dS_t = change in stock price
    mu = drift (expected return)
    S_t = stock price at time t
    dt = time step
    sigma = volatility (standard deviation of returns)
    dW_t = Wiener process (random variable with a normal distribution with mean 0 and variance dt)

The code is written in Python and uses libraries such as numpy for numerical calculations and matplotlib for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
mu, n, T, M, S0, sigma = 0.1, 100, 1, 5000, 100, 0.3

dt = T / n  # time step
St = np.exp((mu - sigma ** 2 / 2) * dt
            + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)  # stock price change with GBM formula

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
