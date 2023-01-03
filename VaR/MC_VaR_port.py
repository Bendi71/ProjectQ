import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stocks = input('Ticker: ').split()
weights = np.array(input('Weights: ').split(), dtype=float)
if weights.size == 0:
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    weights = np.around(weights, 2)


def GetData(stock, time):
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    hozam = hist.pct_change()
    meanreturn = hozam.mean()
    covmatrix = hozam.cov()
    return meanreturn, covmatrix, np.array(hist.tail(1))


def MC(hist, mc, weights, time):
    MeanRmatrix = np.full(shape=(time, len(weights)), fill_value=hist[0]).T
    sim_port = np.full(shape=(time, mc), fill_value=0.0)
    daily_return = np.full(shape=(time, mc), fill_value=0.0)
    initialvalue = np.inner(weights, hist[2])
    for i in range(mc):
        """simulation with cholesky-decomposition
        As the covariance matrix is positive definite, we can use the decomposition
        Returns of portfolio = Mean Returns + L * Z, where L is the cholesky-decomp form of covariance matrix, 
        and Z is a sample matrix of a normal distribution"""
        L = np.linalg.cholesky(hist[1])
        Z = np.random.normal(size=(time, len(weights)))
        sim_returns = MeanRmatrix + np.inner(L, Z)
        daily_return[:, i] = np.inner(weights, sim_returns.T)
        sim_port[:, i] = (1 + np.inner(weights, sim_returns.T)).cumprod() * initialvalue
    port_return = (sim_port[-1] - initialvalue) / initialvalue
    plt.plot(sim_port)
    # plt.show()
    return daily_return, port_return.mean()


def VaR(hozam, alfa):
    return np.percentile(hozam, alfa)


def cVaR(hozam, alfa):
    feltetel = hozam <= VaR(hozam, alfa)
    return np.mean(hozam[feltetel])


time = {'3mo': 63, '1y': 1 * 252, '2y': 2 * 252, '5y': 5 * 252, '10y': 10 * 252}

eredmenyek = []
for i in range(5):
    final = MC(GetData(stocks, list(time.items())[i][0]), 200, weights, list(time.items())[i][1])
    values = pd.Series(final[0][-1, :])
    adatok = []
    adatok.append(list(time.items())[i][0])
    adatok.extend([final[1], values.mean(), values.std(), values.min()])
    adatok.extend([VaR(values, alfa=5), cVaR(values, alfa=5), VaR(values, alfa=10), cVaR(values, alfa=10)])
    eredmenyek.append(adatok)

with pd.option_context('display.max_columns', None):
    print(pd.DataFrame(eredmenyek,
                       columns=['Time', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%', 'cVaR 5%',
                                'VaR 10%', 'cVaR 10%']))
