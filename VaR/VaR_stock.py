import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock = input('Ticker: ')  # Input is a ticker of a stock


def GetData(stock, time):  # Downloads historical stock prices from yahoofinance and does basic statistical calculations
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    returns = np.log(hist / hist.shift(1)).dropna()  # Daily log returns
    mean_return = np.mean(returns)
    sig = np.std(returns)  # Standard deviation is returns related
    maxdraw = np.min(returns)
    Return = np.log((hist[-1] - hist[0]) / hist[0])  # Return of the stock from start to end date
    return returns, Return, mean_return, sig, maxdraw, hist[-1]  # Output is an array of values


def MC(hist, mc, time):  # Based on historical data simulates future stock returns, and therefore prices with Monte
    # Carlo simulation
    sim_returns = np.full(shape=(time, mc), fill_value=0.0)  # Makes a matrix for simulated returns with adequate size
    stock_return = []
    for i in range(mc):
        sim_returns[:, i] = np.random.normal(hist[2], hist[3], time)  # Sample from normal distribution with
        # historical mean and volatility
        sim_price = hist[5] * (1 + sim_returns[:, i]).cumprod()  # Calculates prices from returns
        stock_return = (sim_price[-1] - hist[5]) / hist[5]
        plt.axhline(hist[5], c='k')
        plt.plot(sim_price)
    # plt.show()  # with making the code alive, you can see the simulations in diagrams
    return sim_returns, np.mean(stock_return)  # Output is an array of array and value


# Different time intervals
time = {'3mo': 63, '1y': 1 * 252, '2y': 2 * 252, '5y': 5 * 252, '10y': 10 * 252}


def VaR(returns, alfa):  # Value at risk from returns
    return np.percentile(returns, alfa)  # Output is the maximum percentile of loss with (1-alfa) significance level


def cVaR(returns, alfa):  # Conditional value at risk from returns
    condition = returns <= VaR(returns, alfa)
    return np.mean(returns[condition])  # Output is the mean percentile of loss beyond VaR cutoff point


# Gets results together by different time intervals
results = []
mcresults = []
for i in range(5):
    data = [list(time.items())[i][0]]
    stockvalues = GetData(stock, list(time.items())[i][0])
    stockreturns = stockvalues[0][1:]
    data.extend([stockvalues[1], stockvalues[2], stockvalues[3], stockvalues[4], VaR(stockreturns, alfa=5),
                 cVaR(stockreturns, alfa=5), VaR(stockreturns, alfa=10), cVaR(stockreturns, alfa=10)])
    results.append(data)

    mcdata = []
    mcvalues = MC(stockvalues, 200, list(time.items())[i][1])
    values = pd.Series(mcvalues[0][-1, :])
    mcdata.append(list(time.items())[i][0])
    mcdata.extend([mcvalues[1], values.mean(), values.std(), values.min()])
    mcdata.extend([VaR(values, alfa=5), cVaR(values, alfa=5), VaR(values, alfa=10), cVaR(values, alfa=10)])
    mcresults.append(mcdata)

# Print in DataFrame format
with pd.option_context('display.max_columns', None):
    print(pd.DataFrame(results,
                       columns=['Time', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%', 'cVaR 5%',
                                'VaR 10%', 'cVaR 10%']))
    print(pd.DataFrame(mcresults,
                       columns=['Time', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%', 'cVaR 5%',
                                'VaR 10%', 'cVaR 10%']))
