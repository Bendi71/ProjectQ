import backtrader as bt
import pandas as pd
import datetime
import yfinance as yf


class MultiStockStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('smaperiod', 50),
        ('printlog', False),
    )

    def __init__(self):
        self.inds = dict()
        self.stocks = self.datas[1:]

        for d in self.stocks:
            self.inds[d] = dict()
            self.inds[d]['sma'] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.maperiod)
            self.inds[d]['rsi'] = bt.indicators.RSI(d.close, period=14)

    def next(self):
        for d in self.stocks:
            pos = self.getposition(d).size
            if pos == 0:
                if self.inds[d]['rsi'] < 30:
                    self.buy(data=d, size=1000)
            else:
                if self.inds[d]['rsi'] > 70:
                    self.sell(data=d, size=1000)

            if self.params.printlog:
                print(f'{d._name} close: {d.close[0]:.2f}, rsi: {self.inds[d]["rsi"][0]:.2f}')

    def stop(self):
        if self.params.printlog:
            for d in self.stocks:
                self.log(f'{d._name}, sma: {self.inds[d]["sma"][0]:.2f}, rsi: {self.inds[d]["rsi"][0]:.2f}')


cerebro = bt.Cerebro()

# Get a list of stock symbols to trade
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Load stock data for each symbol and add it to the cerebro instance
for symbol in symbols:
    data = yf.download(symbol, period='1y', interval='1d')
    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data, name=symbol)

cerebro.addstrategy(MultiStockStrategy)

cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

cerebro.broker.set_cash(100000)

cerebro.addobserver(bt.observers.BuySell)

results = cerebro.run()
print(cerebro.broker.getvalue())


