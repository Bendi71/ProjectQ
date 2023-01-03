import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock=input('Ticker: ')
def GetData(stock,time):
    hist=yf.download(tickers=stock,period=time,interval='1d',auto_adjust=True,ignore_tz=True)['Close']
    hozam=hist.pct_change()
    atlag=np.mean(hozam)
    sig=np.std(hozam)
    maxdraw=np.min(hozam)
    osszhozam=(hist[-1]-hist[0])/hist[0]
    return hozam,osszhozam, atlag, sig ,maxdraw, hist[-1]

def MC(hist,mc,time):
    sim_returns=np.full(shape=(time,mc),fill_value=0.0)
    stock_return=[]
    for i in range(mc):
        sim_returns[:,i]=np.random.normal(hist[2],hist[3],time)
        sim_price=hist[5]*(1+sim_returns[:,i]).cumprod()
        stock_return=(sim_price[-1]-hist[5])/hist[5]
        plt.axhline(hist[5],c='k')
        plt.plot(sim_price)
    #plt.show()
    return sim_returns, np.mean(stock_return)

time={'3mo':63,'1y':1*252,'2y':2*252,'5y':5*252,'10y':10*252}

def VaR(hozam,alfa):
    return np.percentile(hozam, alfa)
def cVaR(hozam,alfa):
    feltetel=hozam<=VaR(hozam,alfa)
    return np.mean(hozam[feltetel])

records=[]
mcrecords=[]
for i in range(5):
    adatok=[]
    adatok.append(list(time.items())[i][0])
    values = GetData(stock, list(time.items())[i][0])
    hozam=values[0][1:]
    adatok.extend([values[1],values[2], values[3],values[4],VaR(hozam,alfa=5),cVaR(hozam,alfa=5),VaR(hozam,alfa=10),cVaR(hozam,alfa=10)])
    records.append(adatok)

    mcadat=[]
    final = MC(values, 200, list(time.items())[i][1])
    values = pd.Series(final[0][-1, :])
    mcadat.append(list(time.items())[i][0])
    mcadat.extend([final[1], values.mean(), values.std(), values.min()])
    mcadat.extend([VaR(values, alfa=5), cVaR(values, alfa=5), VaR(values, alfa=10), cVaR(values, alfa=10)])
    mcrecords.append(mcadat)

with pd.option_context('display.max_columns',None):
    print(pd.DataFrame(records,columns=['Time','Return','Mean Return','Deviation','Max Drawdown','VaR 5%','cVaR 5%','VaR 10%','cVaR 10%']))
    print(pd.DataFrame(mcrecords,columns=['Time','Return','Mean Return','Deviation','Max Drawdown','VaR 5%','cVaR 5%','VaR 10%','cVaR 10%']))