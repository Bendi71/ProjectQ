import datetime
import backtrader as bt
from strategies import *
import yfinance as yf

# Instantiate Cerebro engine
cerebro = bt.Cerebro()

# Set data parameters and add to Cerebro
data = hist = yf.download(tickers='TSLA', period='1y', interval='1d', auto_adjust=True, ignore_tz=True)['Close']

cerebro.adddata(data)

# Add strategy to Cerebro
cerebro.addstrategy(AverageTrueRange)

# Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=3)

if __name__ == '__main__':
    # Run Cerebro Engine
    start_portfolio_value = cerebro.broker.getvalue()

    cerebro.run()

    end_portfolio_value = cerebro.broker.getvalue()
    pnl = end_portfolio_value - start_portfolio_value
    print(f'Starting Portfolio Value: {start_portfolio_value:2f}')
    print(f'Final Portfolio Value: {end_portfolio_value:2f}')
    print(f'PnL: {pnl:.2f}')