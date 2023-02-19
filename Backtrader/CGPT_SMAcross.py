import backtrader as bt

class SMACross(bt.Strategy):
    params = (('sma1', 10), ('sma2', 30), ('percent', 0.95))

    def __init__(self):
        self.sma1 = bt.indicators.SMA(self.data, period=self.params.sma1)
        self.sma2 = bt.indicators.SMA(self.data, period=self.params.sma2)
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)
        self.percent = self.params.percent

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = self.broker.getcash() * self.percent / self.data.close
                self.buy(size=size)
        elif self.crossover < 0:
            self.close()