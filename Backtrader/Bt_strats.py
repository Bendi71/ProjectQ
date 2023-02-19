import backtrader as bt


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
            self.bar_executed = len(self)
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
        self.dc = bt.Indicator.DonchianChannels(self.data, period=self.params.period)
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
            if self.data.close[0] > self.dc.lines.top[0]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.dc.lines.bot[0]:
                self.order = self.sell()
