import yfinance as yf
import numpy as np
import pandas as pd

stocks = input('Ticker: ').split()
weights = np.array(input('Weights: ').split(), dtype=float)
if weights.size == 0:
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    weights = np.around(weights, 2)


def GetData(stock, time):
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    hozam = hist.pct_change()
    covmat = hozam.cov()
    Returns = (hist.tail(1).iloc[0] - hist.head(1).iloc[0]) / hist.head(1).iloc[0]
    return hozam.dropna(), Returns, covmat


def PortPerformance(weights, dailyreturns, returns, covmatrix):
    Return = np.sum(weights * returns)
    stdev = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))
    portreturns = dailyreturns.dot(weights)
    mean = portreturns.mean()
    maxdraw = portreturns.min()
    return Return, mean, stdev, maxdraw, portreturns


def VaR(hozam, alfa):
    return np.percentile(hozam, alfa)


def cVaR(hozam, alfa):
    feltetel = hozam <= VaR(hozam, alfa)
    return np.mean(hozam[feltetel])


time = {'3mo': 63, '1y': 1 * 252, '2y': 2 * 252, '5y': 5 * 252, '10y': 10 * 252}

results = []
for i in range(5):
    stockvalues = GetData(stocks, list(time.items())[i][0])
    portvalues = PortPerformance(weights, stockvalues[0], stockvalues[1], stockvalues[2])
    returns = portvalues[4]
    data = [list(time.items())[i][0], weights]
    data.extend([portvalues[0], portvalues[1], portvalues[2], portvalues[3]])
    data.extend([VaR(returns, alfa=5), cVaR(returns, alfa=5), VaR(returns, alfa=10), cVaR(returns, alfa=10)])
    results.append(data)

with pd.option_context('display.max_columns', None):
    print(pd.DataFrame(results,
                       columns=['Time', 'Weights', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%',
                                'cVaR 5%', 'VaR 10%', 'cVaR 10%']))
