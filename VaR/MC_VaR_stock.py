import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock = input('Ticker: ')


def GetData(stock):
    hist = yf.download(tickers=stock, period='2y', interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    hozam = np.log(hist / hist.shift(1)).dropna()  # Daily log returns
    atlag = np.mean(hozam)
    sig = np.std(hozam)
    return atlag, sig, hist[-1]


time = {'3m': 63, '1y': 1 * 252, '2y': 2 * 252, '5y': 5 * 252, '10y': 10 * 252}


def MC(hist, mc, time):
    sim_returns = np.full(shape=(time, mc), fill_value=0.0)
    stock_return = []
    for i in range(mc):
        sim_returns[:, i] = np.random.normal(hist[0], hist[1], time)
        sim_price = hist[2] * (1 + sim_returns[:, i]).cumprod()
        stock_return = (sim_price[-1] - hist[2]) / hist[2]
        plt.axhline(hist[2], c='k')
        plt.plot(sim_price)
    plt.show()
    return sim_returns, np.mean(stock_return)


def VaR(hozam, alfa):
    return np.percentile(hozam, alfa)


def cVaR(hozam, alfa):
    feltetel = hozam <= VaR(hozam, alfa)
    return np.mean(hozam[feltetel])


eredmenyek = []
for i in range(5):
    final = MC(GetData(stock), 200, list(time.items())[i][1])
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
