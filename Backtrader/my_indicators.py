import pandas
import backtrader as bt
import numpy as np

def RSI(df, periods, ema=True):
    close_delta = df['Close'].diff()

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema:
        ma_up = up.ewm(com=periods - 1, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, min_periods=periods).mean()
    else:
        ma_up = up.rolling(window=periods).mean()
        ma_down = down.rolling(window=periods).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def Donchian(df,period):
    upper=df['High'].rolling(period).max()
    lower=df['Low'].rolling(period).min()
    mid=(upper+lower)/2
    return upper,lower,mid

class DonchianChannels(bt.Indicator):
    '''
    Params Note:
      - `lookback` (default: -1)
        If `-1`, the bars to consider will start 1 bar in the past and the
        current high/low may break through the channel.
        If `0`, the current prices will be considered for the Donchian
        Channel. This means that the price will **NEVER** break through the
        upper/lower channel bands.
    '''

    alias = ('DCH', 'DonchianChannel',)

    lines = ('dcm', 'dch', 'dcl',)  # dc middle, dc high, dc low
    params = dict(
        period=20,
        lookback=-1,  # consider current bar or not
    )

    plotinfo = dict(subplot=False)  # plot along with data
    plotlines = dict(
        dcm=dict(ls='--'),  # dashed line
        dch=dict(_samecolor=True),  # use same color as prev line (dcm)
        dcl=dict(_samecolor=True),  # use same color as prev line (dch)
    )

    def __init__(self):
        hi, lo = self.data.high, self.data.low
        if self.p.lookback:  # move backwards as needed
            hi, lo = hi(self.p.lookback), lo(self.p.lookback)

        self.l.dch = bt.ind.Highest(hi, period=self.p.period)
        self.l.dcl = bt.ind.Lowest(lo, period=self.p.period)
        self.l.dcm = (self.l.dch + self.l.dcl) / 2.0  # avg of the above
def MACD(df, fast, slow, signal):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def SMA(df, periods):
    return df.rolling(window=periods).mean()

def EMA(df, periods):
    return df.ewm(span=periods, adjust=False).mean()

def ATR(df, periods):
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return tr.rolling(window=periods).mean()

def StochasticOscillator(df, k, d):
    stochk = ((df['Close'] - df['Low'].rolling(k).min()) / (df['High'].rolling(k).max() - df['Low'].rolling(k).min())) * 100
    stochd = stochk.rolling(d).mean()
    return stochk, stochd

def BollingerBands(df, periods):
    sma = df['Close'].rolling(window=periods).mean()
    rstd = df['Close'].rolling(window=periods).std()
    upper_band = sma + 2 * rstd
    lower_band = sma - 2 * rstd
    return upper_band, lower_band

def OBV(df):
    return ((np.sign(df['Close'].diff()) * df['Volume']).fillna(0)).cumsum()

def VWAP(df):
    vwap = ((df['Volume'] * (df['High'] + df['Low'] + df['Close'])) / 3).cumsum() / df['Volume'].cumsum()
    return vwap