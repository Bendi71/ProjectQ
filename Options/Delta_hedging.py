"""
Delta Hedging Strategy Implementation Project

This project aims to implement a delta hedging strategy for a European call option on a single stock. Delta hedging is a strategy used by options traders to reduce the risk associated with the price movements of the underlying asset.

The strategy involves adjusting the position in the underlying asset (in this case, a stock) to offset changes in the option's delta. The delta of an option measures the rate of change of the option price with respect to changes in the underlying asset's price. By maintaining a delta-neutral position, the trader can mitigate the risk of losses due to unfavorable price movements in the underlying asset.

In this project, we use the Black-Scholes model to calculate the delta of a European call option. The delta is recalculated at regular intervals, and the position in the stock is adjusted accordingly. The cost of these adjustments, including the cost of borrowing funds to purchase the stock, is tracked over time.

The code is written in Python and uses libraries such as yfinance for downloading stock price data, numpy for numerical calculations, pandas for data manipulation, and matplotlib for plotting. The Black-Scholes model is implemented in a separate module and imported into the main script.
"""
import numpy as np
import Black_Scholes_CallPut as bs
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Delta hedging with underlying asset

strike = 300
rf = 0.06
N=5

data = yf.download('MSFT', period='1y', interval='1d', auto_adjust=True, ignore_tz=True)
prices=data.iloc[::N,:]['Close']
period=len(prices)
time_to_expire = [(period - i)/period for i in range(0,period)]
db = 1000

# Historical volatility
close = data['Close']
log_returns = np.log(close / close.shift(1)).dropna()
vol =log_returns.std() * np.sqrt(252)

delta=[bs.Call(prices[i],strike,time_to_expire[i],vol,rf).Delta() for i in range(period)]

#Initial
stock_adj=[delta[0]*db]
cost_of_adj=[stock_adj[0]*prices[0]]
total_cost=[stock_adj[0]*prices[0]]
interest_cost=[total_cost[0]*rf*N/365]

#Rebalancing
for i in range(1,period):
    delta_diff=delta[i]-delta[i-1]
    stock_adj.append(delta_diff*db)
    cost_of_adj.append(stock_adj[i]*prices[i])
    total_cost.append(total_cost[i-1]+cost_of_adj[i]+interest_cost[i-1])
    interest_cost.append(total_cost[i]*rf*N/365)

optionprice=bs.Call(prices[0],strike,time_to_expire[0],vol,rf).CallPrice() * db


option_pnl= optionprice - (total_cost[-1]+interest_cost[-1])
print(option_pnl)
pd.set_option('display.max_columns', 7)
sim = pd.DataFrame({'Stock Price':prices, 'Delta':np.round(delta, 2),'Shares Purchased':np.array(stock_adj),
                    'Cost of Shares':np.round(cost_of_adj,-2),
                    'Cumulative Cost': total_cost,
                    'Interest Cost':np.array(interest_cost)/1000})
print(sim)
