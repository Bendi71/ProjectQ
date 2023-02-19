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
    dailyreturns = hist.pct_change()
    meanreturn = dailyreturns.mean()
    covmatrix = dailyreturns.cov()  # Covariance matrix
    Returns = (hist.tail(1).iloc[0] - hist.head(1).iloc[0]) / hist.head(1).iloc[0]  # Return of the stocks from start
    # to end date
    return Returns, covmatrix  # Output is an array of arrays


def PortPerformance(weights, returns, covmatrix):  # Calculates performance of the portfolio, such as
    Return = np.sum(weights * returns)  # Overall return
    stdev = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))  # Standard deviation of portfolio
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

#def OptimalPortfolio(results, covmatrix, riskfreerate = 0,):


ret, covmat = GetData(stocks,'2y')

maxSRret, maxSRvar = PortPerformance(maxSharpeRatio(ret,covmat).x,ret,covmat)
minVarret, minVarvar= PortPerformance(minPortfolioVariance(ret,covmat).x, ret, covmat)
targetreturns=np.linspace(minVarret,maxSRret,10)
efficientList = []
for target in targetreturns:
    data=[target,OptimalPortfolio(ret, covmat, target).fun]
    efficientList.append(data)

frontier=pd.DataFrame(efficientList,columns=['Return','Volatility'])
plt.scatter(x=frontier['Volatility'],y=frontier['Return'])
plt.show()