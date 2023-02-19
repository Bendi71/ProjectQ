import backtrader as bt
import backtrader.indicators as btind

# Define the RSI strategy class
class RSIStrategy(bt.Strategy):
    params = (('period', 14), ('upper', 70), ('lower', 30), ('printlog', False))

    def __init__(self):
        self.rsi = btind.RSI_SMA(self.data, period=self.params.period)
        self.upper = self.params.upper
        self.lower = self.params.lower
        self.sell_sig = btind.CrossDown(self.rsi, self.upper)
        self.buy_sig = btind.CrossUp(self.rsi, self.lower)
        self.dataclose = self.data.close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.inds = dict()

    def log(self, txt, dt=None):
        if self.params.printlog or dt is None:
            print(txt)
        else:
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                order.executed.price, order.executed.value, order.executed.comm))
            self.bar_executed = len(self)
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.buy_sig[0] == 1:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.sell_sig[0] == 1:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()