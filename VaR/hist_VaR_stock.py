import yfinance as yf
import numpy as np
import pandas as pd

stock = input('Ticker: ')


def GetData(stock, time):
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    hozam = hist.pct_change()
    atlag = np.mean(hozam)
    var = np.std(hozam)
    maxdraw = np.min(hozam)
    osszhozam = (hist[-1] - hist[0]) / hist[0]
    return hozam, osszhozam, atlag, var, maxdraw


def VaR(hozam, alfa):
    return np.percentile(hozam, alfa)


def cVaR(hozam, alfa):
    feltetel = hozam <= VaR(hozam, alfa)
    return np.mean(hozam[feltetel])


time = ['3mo', '1y', '2y', '5y', '10y']
eredmenyek = []
for t in time:
    adatok = []
    adatok.append(t)
    values = GetData(stock, t)
    hozam = values[0][1:]
    for i in range(1, 5):
        adatok.append(values[i])
    adatok.append(VaR(hozam, alfa=5))
    adatok.append(cVaR(hozam, alfa=5))
    adatok.append(VaR(hozam, alfa=10))
    adatok.append(cVaR(hozam, alfa=10))
    eredmenyek.append(adatok)

with pd.option_context('display.max_columns', None):
    print(pd.DataFrame(eredmenyek,
                       columns=['Time', 'Return', 'Mean Return', 'Deviation', 'Max Drawdown', 'VaR 5%', 'cVaR 5%',
                                'VaR 10%', 'cVaR 10%']))
