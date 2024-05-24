"""
Backtrader Stock Frame Project

This project aims to implement various trading strategies using the Backtrader framework. Backtrader is a Python library used for backtesting trading strategies.

The script takes a ticker symbol, start date, and end date as inputs from the user. It then downloads the historical stock price data for the given ticker and date range using the yfinance library.

The user is also prompted to choose a trading strategy from a list of strategies defined in the `Bt_strats.py` file. The chosen strategy is then applied to the historical stock price data.

The script uses the Backtrader library to set up a trading environment, add the historical data, and add the chosen strategy. It also sets the initial cash, commission, stake size, and slippage for the trading environment.

The script then runs the backtest and prints the starting and final portfolio values. It also calculates and prints various performance metrics, such as total return, annual return, Sharpe ratio, maximum drawdown, total trades, total wins, total losses, and win rate.

Finally, the script plots the buy/sell signals and closing price of the stock, along with any additional plots specific to the chosen strategy.

The code is written in Python and uses libraries such as yfinance for downloading stock price data, Backtrader for backtesting trading strategies, and matplotlib for plotting.
"""
import datetime
import re

import my_indicators as ind
import backtrader as bt
import yfinance as yf
import Bt_strats as st
import matplotlib.pyplot as plt

# get strategy names from Bt_strats.py
strat_names = re.compile(r'class\s+([^\(\s]+)\s*[:\(]').findall(open('Bt_strats.py', 'r').read())

# inputs
ticker = input("Ticker: ")
startdate = input("Start date: ")
closedate = input("End date: ")

print("Choose an option:")
for i, strat in enumerate(strat_names, 1):
    print(f"{i}. {strat}")
user_strat = input("Enter the number corresponding to your choice: ")

# Validate user input
try:
    user_choice = int(user_strat)
    if 1 <= user_choice <= len(strat_names):
        chosen_option = strat_names[user_choice - 1]
        print(f"You chose: {chosen_option}")
    else:
        print("Invalid choice. Please enter a valid number.")
except ValueError:
    print("Invalid choice. Please enter a valid number.")

cerebro = bt.Cerebro()

# add data feed
df = yf.download(ticker, start=startdate, end=closedate, interval='1d', auto_adjust=True, ignore_tz=True)
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# add strategy
strat = st.__dict__[chosen_option]
cerebro.addstrategy(strat)

# Set the cash, commission, stake size, slippage
cerebro.broker.setcash(1)
cerebro.broker.setcommission(commission=0.001)
cerebro.addsizer(bt.sizers.PercentSizer, percents=20)
cerebro.broker.set_slippage_fixed(0.01, slip_open=True, slip_match=True, slip_out=True)

# Add performance analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

# Run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# calculate performance metrics
returns = cerebro.broker.getvalue() / cerebro.broker.startingcash - 1
annual_returns = pow(1 + returns, 252 / len(data)) - 1
sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()['sharperatio']
drawdown = results[0].analyzers.drawdown.get_analysis()
trade_analysis = results[0].analyzers.trade_analyzer.get_analysis()
total_trades = trade_analysis['total']['total']
total_wins = trade_analysis['won']['total']
total_losses = trade_analysis['lost']['total']
win_rate = (total_wins / total_trades) * 100

# Print performance metrics
print('Total Return:', round(returns * 100, 2), '%')
print('Annual Return:', round(annual_returns * 100, 2), '%')
print('Sharpe Ratio: %.2f' % sharpe_ratio)
print('Max Drawdown: %.2f%%' % drawdown['max']['drawdown'])
print('Total Trades:', total_trades)
print('Total Wins:', total_wins)
print('Total Losses:', total_losses)
print('Win Rate: %.2f%%' % win_rate)

# Plotting
dates = [datetime.date.fromordinal(int(tim)) for tim in data.datetime.array]
buy_times = strat.buy_times
sell_times = strat.sell_times
closes = data.close.array

if strat == st.RSIStrategy:
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

    # First subplot for buy/sell signals and closing price
    ax1.scatter(buy_times, strat.buy_prices, marker='^', color='green', label='Buy', s=100)
    ax1.scatter(sell_times, strat.sell_prices, marker='v', color='red', label='Sell', s=100)
    ax1.plot(dates, closes, color='black', alpha=0.5)
    ax1.set_ylabel('Close Price')
    ax1.set_title('MSFT')

    # Second subplot for RSI
    rsi = ind.rsi(df['Close'], periods=10)
    ax2.plot(dates, rsi, color='blue')
    ax2.axhline(65, color='red', linestyle='dotted')
    ax2.axhline(35, color='green', linestyle='dotted')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
elif strat == st.SMACrossStrategy:
    plt.plot(dates, closes, color='black', alpha=0.5)
    plt.ylabel('Close Price')
    plt.title('MSFT')
    plt.scatter(buy_times, strat.buy_prices, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_times, strat.sell_prices, marker='v', color='red', label='Sell', s=100)
    plt.plot(dates, df['Close'].rolling(window=30).mean(), color='blue')
    plt.plot(dates, df['Close'].rolling(window=10).mean(), color='red')
elif strat == st.DonchianBreakout:
    plt.axes().set_facecolor('black')
    plt.plot(dates, closes, color='white', alpha=0.5)
    plt.ylabel('Close Price')
    plt.title('MSFT')
    plt.scatter(buy_times, strat.buy_prices, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_times, strat.sell_prices, marker='v', color='red', label='Sell', s=100)
    plt.plot(dates, ind.donchianchannel(df, 20)['upper'], color='darkcyan')
    plt.plot(dates, ind.donchianchannel(df, 20)['lower'], color='darkcyan')
    plt.plot(dates, ind.donchianchannel(df, 20)['mid'], color='lightcyan')
elif strat == st.TrendFollowingStrategy:
    plt.plot(dates, closes, color='black', alpha=0.5)
    plt.ylabel('Close Price')
    plt.title('MSFT')
    plt.scatter(buy_times, strat.buy_prices, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_times, strat.sell_prices, marker='v', color='red', label='Sell', s=100)
elif strat == st.MomentumStrategy:
    plt.plot(dates, closes, color='black', alpha=0.5)
    plt.ylabel('Close Price')
    plt.title('MSFT')
    plt.scatter(buy_times, strat.buy_prices, marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_times, strat.sell_prices, marker='v', color='red', label='Sell', s=100)
plt.show()
