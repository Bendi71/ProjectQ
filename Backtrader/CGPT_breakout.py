import backtrader as bt

class BreakoutStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),
        ('entry_multiple', 1.5),
        ('exit_multiple', 0.5)
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.entry_price = 0
        self.stop_price = 0
        self.take_profit_price = 0

    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.high[-2]:
                self.entry_price = self.data.close[0]
                self.stop_price = self.entry_price - (self.atr[0] * self.params.entry_multiple)
                self.take_profit_price = self.entry_price + (self.atr[0] * self.params.entry_multiple)
                self.buy(price=self.entry_price, exectype=bt.Order.Stop)
                self.sell(price=self.stop_price, exectype=bt.Order.Stop, size=self.position.size,
                          oco=self.take_profit_price)

        else:
            if self.data.close[0] < self.stop_price:
                self.close()
            elif self.data.close[0] > self.take_profit_price:
                self.close()


"""Explanation 
The params attribute is a tuple that contains default parameter values for the strategy. In this case, 
the atr_period parameter sets the lookback period for the Average True Range (ATR) indicator, and the entry_multiple 
and exit_multiple parameters set the multiple of the ATR used for setting the entry and exit prices. 

The __init__ method is called when an instance of the strategy is created. In this method, we initialize the ATR 
indicator using the bt.indicators.ATR function provided by backtrader. We also initialize the entry_price, 
stop_price, and take_profit_price variables to 0. 

The next method is called for each data point in the data feed. In this method, we implement the trading logic for 
the strategy. 

The first if statement checks if we do not have a position in the market. If this is the case, we check if the 
current closing price is greater than the high of the previous period. If it is, we initiate a long position by 
setting the entry_price, stop_price, and take_profit_price variables. We then place a buy order at the entry_price 
using the bt.Order.Stop order type. We also place a sell order at the stop_price with the oco parameter set to the 
take_profit_price. This creates an OCO (one cancels the other) order, which means that if one order is executed, 
the other order will be canceled. 

The second else statement is executed if we have an open position in the market. If this is the case, we check if the 
current closing price is less than the stop_price. If it is, we close the position using the close method. If the 
current closing price is greater than the take_profit_price, we also close the position using the close method. """
