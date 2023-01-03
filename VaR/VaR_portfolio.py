import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stocks = input('Ticker: ').split()  # Input is tickers of stocks
weights = np.array(input('Weights: ').split(), dtype=float)  # You can type in weights based on your portfolio
if weights.size == 0:
    weights = np.random.random(len(stocks))  # Or you can continue with random weights
    weights /= np.sum(weights)
    weights = np.around(weights, 2)


def GetData(stock, time):  # Downloads historical stock prices from yahoofinance and does basic statistical calculations
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    hozam = hist.pct_change()
    meanreturn = hozam.mean()
    covmatrix = hozam.cov()  # Covariance matrix
    Returns = (hist.tail(1).iloc[0] - hist.head(1).iloc[0]) / hist.head(1).iloc[0]  # Return of the stocks from start
    # to end date
    return hozam.dropna(), Returns, meanreturn, covmatrix, np.array(hist.tail(1))  # Output is an array of arrays


def PortPerformance(weights, dailyreturns, returns, covmatrix):  # Calculates performance of the portfolio, such as
    Return = np.sum(weights * returns)  # Overall return
    stdev = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))  # Standard deviation of portfolio
    portreturns = dailyreturns.dot(weights)  # Daily returns
    mean = portreturns.mean()  # Mean of daily returns
    maxdraw = portreturns.min()  # Maximum one-day drawdown
    return Return, mean, stdev, maxdraw, portreturns  # Output is an array of values and array


def MC(hist, mc, weights, time):  # Based on historical data simulates future portfolio returns, and therefore value
    # with Monte Carlo simulation
    MeanRmatrix = np.full(shape=(time, len(weights)), fill_value=hist[2]).T  # Matrix of mean daily returns
    sim_port = np.full(shape=(time, mc), fill_value=0.0)  # Sets up matrix for simulated portfolio value
    daily_return = np.full(shape=(time, mc), fill_value=0.0)  # Sets up matrix for simulated daily returns
    initialvalue = np.inner(weights, hist[4])  # Initial value of portfolio is the sumproduct of weights by last price
    for i in range(mc):
        """simulation with Cholesky-decomposition
        As the covariance matrix is positive definite, we can use the decomposition
        Returns of portfolio = Mean Returns + L * Z, where L is the Cholesky-decomposition form of covariance matrix, 
        and Z is a matrix of samples from normal distribution"""
        L = np.linalg.cholesky(hist[3])
        Z = np.random.normal(size=(time, len(weights)))
        sim_returns = MeanRmatrix + np.inner(L, Z)
        daily_return[:, i] = np.inner(weights, sim_returns.T)  # Simulates daily returns
        sim_port[:, i] = (1 + daily_return[:, i]).cumprod() * initialvalue  # Simulates daily value of portfolio
    port_return = (sim_port[-1] - initialvalue) / initialvalue  # Calculates overall return of portfolio
    plt.plot(sim_port)
    # plt.show()  # with making the code alive, you can see the simulations in diagrams
    return daily_return, port_return.mean()  # Output is an array and a value


def VaR(returns, alfa):  # Value at risk from returns
    return np.percentile(returns, alfa)  # Output is the maximum percentile of loss with (1-alfa) significance level


def cVaR(returns, alfa):  # Conditional value at risk from returns
    condition = returns <= VaR(returns, alfa)
    return np.mean(returns[condition])  # Output is the mean percentile of loss beyond VaR cutoff point


# Different time intervals
time = {'3mo': 63, '1y': 1 * 252, '2y': 2 * 252, '5y': 5 * 252, '10y': 10 * 252}

# Gets data together by different time intervals
results = []
mcresults = []
for i in range(5):
    stockvalues = GetData(stocks, list(time.items())[i][0])
    portvalues = PortPerformance(weights, stockvalues[0], stockvalues[1], stockvalues[3])
    mcvalues = MC(stockvalues, 200, weights, list(time.items())[i][1])

    returns = portvalues[4]
    data = [list(time.items())[i][0], weights]
    data.extend([portvalues[0], portvalues[1], portvalues[2], portvalues[3]])
    data.extend([VaR(returns, alfa=5), cVaR(returns, alfa=5), VaR(returns, alfa=10), cVaR(returns, alfa=10)])
    results.append(data)

    mcdata = []
    values = pd.Series(mcvalues[0][-1, :])
    mcdata.append(list(time.items())[i][0])
    mcdata.extend([mcvalues[1], values.mean(), values.std(), values.min()])
    mcdata.extend([VaR(values, alfa=5), cVaR(values, alfa=5), VaR(values, alfa=10), cVaR(values, alfa=10)])
    mcresults.append(mcdata)

# Print in DataFrame format
with pd.option_context('display.max_columns', None):
    print(pd.DataFrame(results,
                       columns=['Time', 'Weights', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%',
                                'cVaR 5%', 'VaR 10%', 'cVaR 10%']))
    print(pd.DataFrame(mcresults,
                       columns=['Time', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%', 'cVaR 5%',
                                'VaR 10%', 'cVaR 10%']))
