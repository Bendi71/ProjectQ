"""
Black-Scholes Option Pricing Model and Greeks Calculation Project

This project aims to implement the Black-Scholes model for option pricing and calculate the Greeks for a given option.

The Black-Scholes model is a mathematical model used to calculate the theoretical price of options. It was developed by economists Fischer Black and Myron Scholes, with contributions from Robert Merton. The model assumes that financial markets are efficient and that the returns of the underlying asset are normally distributed.

The Black-Scholes equation is a partial differential equation which describes the price of the option over time. The solution to this equation is the Black-Scholes formula, which gives the price of a European call or put option:

    C = S*N(d1) - X*e^(-rt)*N(d2) for a call option
    P = X*e^(-rt)*N(-d2) - S*N(-d1) for a put option

where:
    C = Call option price
    P = Put option price
    S = Current price of the underlying asset
    X = Strike price of the option
    r = Risk-free interest rate
    t = Time to expiration
    N = Cumulative distribution function of the standard normal distribution
    d1 = (ln(S/X) + (r + (v^2)/2)*t) / (v*sqrt(t))
    d2 = d1 - v*sqrt(t)

The Greeks are financial measures that describe the sensitivity of an option's price to various factors, such as changes in the underlying asset price, volatility, time decay, and interest rate. They are used by options traders to manage the risk and potential rewards of options.

In this project, we calculate the following Greeks:

    Delta: Measures the rate of change of the option price with respect to changes in the underlying asset's price.
    Gamma: Measures the rate of change of Delta with respect to changes in the underlying asset's price.
    Vega: Measures the rate of change of the option price with respect to changes in the underlying asset's volatility.
    Theta: Measures the rate of change of the option price with respect to the passage of time.
    Rho: Measures the rate of change of the option price with respect to changes in the risk-free interest rate.

The code is written in Python and uses libraries such as numpy for numerical calculations and scipy for statistical functions.
"""
import numpy as np
from scipy.stats import norm as sc


class Call:
    def __init__(self, asset_price, strike_price, time_toexpiration, volatility, riskfreerate):
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.time_ex = time_toexpiration
        self.sigma = volatility
        self.rf = riskfreerate
        self.P = np.exp(-self.rf * self.time_ex)
        if self.time_ex != 0:
            self.L = self.sigma * self.time_ex ** 0.5
        self.d1 = np.log(self.asset_price / (self.strike_price * self.P)) / self.L + 0.5 * self.L
        self.d2 = self.d1 - self.L

    def CallPrice(self):
        callprice = self.asset_price * sc.cdf(self.d1) - self.strike_price * self.P * sc.cdf(self.d2)
        return callprice

    def Delta(self):
        return sc.cdf(self.d1)

    def Gamma(self):
        return sc.pdf(self.d1 / (self.asset_price * self.L))

    def Vega(self):
        return self.asset_price * sc.pdf(self.d1) * np.sqrt(self.time_ex) / 100

    def Theta(self):
        return (-self.asset_price * sc.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.time_ex)) - self.rf *
                self.strike_price * self.P * sc.cdf(self.d2)) / 365

    def Rho(self):
        return -self.time_ex * self.strike_price * self.P * sc.cdf(self.d2) / 100


class Put:
    def __init__(self, asset_price, strike_price, time_toepiration, volatility, riskfreerate):
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.time_ex = time_toepiration
        self.sigma = volatility
        self.rf = riskfreerate
        self.P = np.exp(-self.rf * self.time_ex)
        self.L = self.sigma * self.time_ex ** 0.5
        self.d1 = np.log(self.asset_price / (self.strike_price * self.P)) / self.L + 0.5 * self.L
        self.d2 = self.d1 - self.L

    def CallPrice(self):
        putprice = self.asset_price * sc.cdf(self.d1 - 1) - self.strike_price * self.P * sc.cdf(1 - self.d2)
        return putprice

    def Delta(self):
        return sc.cdf(self.d1 - 1)

    def Gamma(self):
        return sc.pdf(self.d1 / (self.asset_price * self.L))

    def Vega(self):
        return self.asset_price * sc.pdf(self.d1) * np.sqrt(self.time_ex) / 100

    def Theta(self):
        return (-self.asset_price * sc.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.time_ex)) + self.rf *
                self.strike_price * self.P * sc.cdf(-self.d2)) / 365

    def Rho(self):
        return -self.time_ex * self.strike_price * self.P * sc.cdf(-self.d2) / 100
