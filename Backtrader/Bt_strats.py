import backtrader as bt
from my_indicators import DonchianChannels


class SMACrossStrategy(bt.Strategy):
    params = (
        ("sma_period_fast", 10),
        ("sma_period_slow", 30),
    )

    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    def __init__(self):
        self.order = None
        self.sma_fast = bt.indicators.SMA(period=self.params.sma_period_fast)
        self.sma_slow = bt.indicators.SMA(period=self.params.sma_period_slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        elif self.crossover < 0:
            self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_times.append(self.datas[0].datetime.datetime())
                self.buy_prices.append(order.executed.price)
            else:
                self.sell_times.append(self.datas[0].datetime.datetime())
                self.sell_prices.append(order.executed.price)
        self.order = None


class RSIStrategy(bt.Strategy):
    params = (('period', 10), ('upper', 65), ('lower', 35), ('printlog', True))

    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data, period=self.params.period)
        self.upper = self.params.upper
        self.lower = self.params.lower
        self.sell_sig = bt.indicators.CrossDown(self.rsi, self.upper)
        self.buy_sig = bt.indicators.CrossUp(self.rsi, self.lower)
        self.order = None
        self.inds = dict()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_times.append(self.datas[0].datetime.datetime())
                self.buy_prices.append(order.executed.price)
            else:
                self.sell_times.append(self.datas[0].datetime.datetime())
                self.sell_prices.append(order.executed.price)
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.buy_sig[0] == 1:
                self.order = self.buy()
        else:
            if self.sell_sig[0] == 1:
                self.order = self.sell()


class DonchianBreakout(bt.Strategy):
    params = (('period', 20), ('printlog', True))

    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    def __init__(self):
        self.dc = DonchianChannels(self.data, period=20)
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_times.append(self.datas[0].datetime.datetime())
                self.buy_prices.append(order.executed.price)
            else:
                self.sell_times.append(self.datas[0].datetime.datetime())
                self.sell_prices.append(order.executed.price)
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.data.close[0] > self.dc.l.dch[0]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.dc.l.dcl[0]:
                self.order = self.sell()


class TrendFollowingStrategy(bt.Strategy):
    params = (('period', 10), ('printlog', True))

    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    def __init__(self):
        self.order = None
        self.period = self.params.period

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_times.append(self.datas[0].datetime.datetime())
                self.buy_prices.append(order.executed.price)
            else:
                self.sell_times.append(self.datas[0].datetime.datetime())
                self.sell_prices.append(order.executed.price)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.data.close[-self.period]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.data.close[-self.period]:
                self.order = self.sell()


class MomentumStrategy(bt.Strategy):
    params = (
        ('sma_period', 50),
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('printlog', True)
    )

    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data, period=self.params.sma_period)
        self.rsi = bt.indicators.RSI(self.data, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data, period_me1=self.params.macd_fast, period_me2=self.params.macd_slow,
                                       period_signal=self.params.macd_signal)
        self.buy_sig = bt.indicators.CrossUp(self.data, self.sma)
        self.sell_sig = bt.indicators.CrossDown(self.data, self.sma)
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_times.append(self.datas[0].datetime.datetime())
                self.buy_prices.append(order.executed.price)
            else:
                self.sell_times.append(self.datas[0].datetime.datetime())
                self.sell_prices.append(order.executed.price)
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.buy_sig[0] == 1 and self.rsi < self.params.rsi_upper and self.macd.macd[0] > self.macd.signal[0]:
                self.order = self.buy()
        else:
            if self.sell_sig[0] == 1 and self.rsi > self.params.rsi_lower and self.macd.macd[0] < self.macd.signal[0]:
                self.order = self.sell()
