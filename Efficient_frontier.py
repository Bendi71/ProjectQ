"""
Efficient Frontier Simulation Project

This project aims to simulate the efficient frontier of a portfolio of stocks. The efficient frontier is a concept in modern portfolio theory which refers to a set of optimal portfolios that offer the highest expected return for a defined level of risk. It is visualized as a curve on a graph where the x-axis represents risk (usually standard deviation of the portfolio's returns) and the y-axis represents expected return. Portfolios that lie on the efficient frontier are considered optimal because they offer the most return for a given level of risk.

The efficient frontier is calculated using historical stock price data and the concept of portfolio diversification. The idea is that by combining different assets, the overall portfolio risk can be reduced without sacrificing expected return. This is due to the correlation between the assets - when some assets are down, others might be up.

The project uses historical stock price data downloaded from Yahoo Finance and calculates the daily log returns and covariance matrix of the returns. It then uses these to calculate the overall return and standard deviation (volatility) of a portfolio for a given set of weights.

The project also includes functions to optimize the portfolio for the maximum Sharpe Ratio (which measures the performance of an investment compared to a risk-free asset, after adjusting for its risk) and minimum variance.

Finally, it simulates the efficient frontier by calculating the portfolio variance for a range of target returns, and plots the efficient frontier, along with the portfolios with the maximum Sharpe Ratio and minimum variance.

The code is written in Python and uses libraries such as yfinance for downloading stock price data, numpy for numerical calculations, pandas for data manipulation, scipy for optimization, and matplotlib for plotting.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scmin

stocks = input('Ticker: ').split()  # Input is tickers of stocks
weights = np.array(input('Weights: ').split(), dtype=float)  # You can type in weights based on your portfolio
if weights.size == 0:
    weights = np.random.random(len(stocks))  # Or you can continue with random weights
    weights /= np.sum(weights)
    weights = np.around(weights, 2)


def GetData(stock, time):  # Downloads historical stock prices from yahoofinance and does basic statistical calculations
    hist = yf.download(tickers=stock, period=time, interval='1d', auto_adjust=True, ignore_tz=True)['Close']
    dailyreturns = np.log(hist / hist.shift(1)).dropna()  # Daily log returns
    meanreturn = dailyreturns.mean()
    covmatrix = dailyreturns.cov()  # Covariance matrix
    Returns = np.log((hist.tail(1).iloc[0] - hist.head(1).iloc[0]) / hist.head(1).iloc[0])  # Return of the stocks from
    return meanreturn, covmatrix  # Output is an array of arrays

def PortPerformance(weights, returns, covmatrix):  # Calculates performance of the portfolio, such as
    Return = np.sum(weights * returns)*252  # yearly mean return
    stdev = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))*np.sqrt(252) # Standard deviation of portfolio
    return Return, stdev  # Output is an array of values

def portReturn(weights, returns, covmatrix):
    return PortPerformance(weights,returns,covmatrix)[0]

def portVariance(weights, returns, covmatrix):
    return PortPerformance(weights,returns,covmatrix)[1] # Standard deviation of portfolio


# We use SLSQP Algorithm for Sharpe-ratio optimization
def negSharpeRatio(weights, returns, covmatrix, riskfreerate = 0):
    portreturn, stdev= PortPerformance(weights,returns,covmatrix)
    return -(portreturn-riskfreerate)/stdev

def maxSharpeRatio(returns, covmatrix, riskfreerate = 0):
    constraint_set=(0,1)
    numofAssets=len(returns)
    maxsp=scmin(fun=negSharpeRatio, x0=weights,args=(returns,covmatrix,riskfreerate),
                bounds=tuple(constraint_set for asset in range(numofAssets)),
                constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
    return maxsp


def minPortfolioVariance(returns, covmatrix):
    constraint_set = (0, 1)
    numofAssets = len(returns)
    minvar = scmin(fun=portVariance, x0=weights, args=(returns, covmatrix),
                  bounds=tuple(constraint_set for asset in range(numofAssets)),
                  constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}), method='SLSQP')
    return minvar

def OptimalPortfolio(returns, covmatrix, targetreturn):
    constraint_set = (0, 1)
    numofAssets = len(returns)
    constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                     {'type': 'eq', 'fun': lambda x: portReturn(x,returns,covmatrix)-targetreturn})
    minvar = scmin(fun=portVariance, x0=weights, args=(returns, covmatrix),
                   bounds=tuple(constraint_set for asset in range(numofAssets)),
                   constraints=constraint, method='SLSQP')
    return minvar

ret, covmat = GetData(stocks,'ytd')

maxSRret, maxSRvar = PortPerformance(maxSharpeRatio(ret,covmat).x,ret,covmat)
minVarret, minVarvar= PortPerformance(minPortfolioVariance(ret,covmat).x, ret, covmat)
targetreturns=np.linspace(minVarret,maxSRret,80)
efficientList = []
for target in targetreturns:
    data=[target,OptimalPortfolio(ret, covmat, target).fun]
    efficientList.append(data)

frontier=pd.DataFrame(efficientList,columns=['Return','Volatility'])
plt.scatter(x=frontier['Volatility'],y=frontier['Return'])
plt.scatter(x=maxSRvar,y=maxSRret,c='red',marker='o',s=100)
plt.scatter(x=minVarvar,y=minVarret,c='green',marker='o',s=100)
plt.legend(['Efficient Frontier','Max Sharpe Ratio','Min Variance'])
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')

plt.show()