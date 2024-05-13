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
