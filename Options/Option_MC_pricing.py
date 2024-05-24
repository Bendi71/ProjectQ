"""
Monte Carlo Simulation for Option Pricing Project

This project aims to implement the Monte Carlo (MC) simulation method for option pricing. The MC simulation is a computational algorithm that relies on repeated random sampling to obtain numerical results. It is often used when the model is complex and an analytical solution is not available.

In the context of option pricing, the MC simulation is used to simulate the possible paths of the underlying asset price over time. The payoff of the option is then calculated for each simulated path. The average payoff, discounted back to the present, gives the price of the option.

The MC simulation is particularly useful for pricing American options and exotic options, where no analytical solutions exist due to the path-dependency and early exercise features.

In this project, we implement the MC simulation for various types of options, including Asian options, Lookback options, Barrier options, and Exchangeable options. For each type of option, we simulate a large number of asset price paths, calculate the payoff for each path, and then take the average and discount it back to the present to get the option price.

The code is written in Python and uses libraries such as numpy for numerical calculations and random number generation.
"""
import numpy as np
from GBM_sim import GBM_sim

class Asian_option:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, steps, time_step, MC_sim):
        self.MC = MC_sim
        self.time_step = time_step
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N

    def asian_call(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            avg_times = np.arange(self.T, 0, -self.T / self.time_step)
            close_step = [min(range(self.N), key=lambda x: abs(x * self.dt - i)) for i in avg_times]
            ST_avg = [ST[j - 1] for j in close_step[1:]]
            ST_avg.append(ST[close_step[0]])
            ST_avg = np.mean(ST_avg)
            payoff = payoff + max(ST_avg - self.K, 0) / self.MC
        return payoff * np.exp(-self.r * self.T)

    def asian_put(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            avg_times = np.arange(self.T, 0, -self.T / self.time_step)
            close_step = [min(range(self.N), key=lambda x: abs(x * self.dt - i)) for i in avg_times]
            ST_avg = [ST[j - 1] for j in close_step[1:]]
            ST_avg.append(ST[close_step[0]])
            ST_avg = np.mean(ST_avg)
            payoff = payoff + max(self.K - ST_avg, 0) / self.MC
        return payoff * np.exp(-self.r * self.T)


class Lookback_option:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, steps, MC_sim):
        self.MC = MC_sim
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N

    def lookback_call(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            payoff = payoff + max(max(ST) - self.K, 0) / self.MC
        return payoff * np.exp(-self.r * self.T)

    def lookback_put(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            payoff = payoff + max(self.K - min(ST), 0) / self.MC
        return payoff * np.exp(-self.r * self.T)

    def lookback_float_call(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            payoff = payoff + max(max(ST) - min(ST), 0) / self.MC
        return payoff * np.exp(-self.r * self.T)

    def lookback_float_put(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            payoff = payoff + max(max(ST) - min(ST), 0) / self.MC
        return payoff * np.exp(-self.r * self.T)


class Barrier_option_MC:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, steps, MC_sim, barrier, barrier_type):
        self.MC = MC_sim
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.bar = barrier
        self.bar_type = barrier_type

    def barrier_call(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            if self.bar_type == 'up':
                if max(ST) > self.bar:
                    payoff = payoff + max(ST[-1] - self.K, 0) / self.MC
                else:
                    payoff = payoff + 0
            elif self.bar_type == 'down':
                if min(ST) < self.bar:
                    payoff = payoff + max(ST[-1] - self.K, 0) / self.MC
                else:
                    payoff = payoff + 0
        return payoff * np.exp(-self.r * self.T)

    def barrier_put(self):
        payoff = 0
        for mc in range(self.MC):
            ST = GBM_sim(self.S, self.r, self.sig, self.T, self.N)
            if self.bar_type == 'up':
                if max(ST) > self.bar:
                    payoff = payoff + max(self.K - ST[-1], 0) / self.MC
                else:
                    payoff = payoff + 0
            elif self.bar_type == 'down':
                if min(ST) < self.bar:
                    payoff = payoff + max(self.K - ST[-1], 0) / self.MC
                else:
                    payoff = payoff + 0
        return payoff * np.exp(-self.r * self.T)

class Exchangeable_options_MC:
    def __init__(self, asset_price_A,asset_price_B,MC_sim,riskfreerate,volatility_A,volatility_B,correlation,
                 time_toexpiration,steps):
        self.MC = MC_sim
        self.SA = asset_price_A
        self.SB = asset_price_B
        self.r = riskfreerate
        self.sigA = volatility_A
        self.sigB = volatility_B
        self.rho = correlation
        self.T = time_toexpiration
        self.N = steps
        self.dt = self.T / self.N
    def call(self):
        payoff=0
        for mc in range(self.MC):
            SA_temp = self.SA
            SB_temp = self.SB
            for t in range(self.N):
                W1 = np.random.standard_normal()
                W2 = np. random.standard_normal() * (1-self.rho**2)**0.5 + W1 * self.rho
                SA_temp = SA_temp + self.r*SA_temp*self.dt+self.sigA*SA_temp*self.dt*W1
                SB_temp = SB_temp + self.r * SB_temp * self.dt + self.sigB * SB_temp * self.dt * W2
            payoff = payoff + max(0,SA_temp-SB_temp)/self.MC
        return payoff * np.exp(-self.r*self.T)

    def put(self):
        payoff = 0
        for mc in range(self.MC):
            SA_temp = self.SA
            SB_temp = self.SB
            for t in range(self.N):
                W1 = np.random.standard_normal()
                W2 = np.random.standard_normal() * (1 - self.rho ** 2) ** 0.5 + W1 * self.rho
                SA_temp = SA_temp + self.r * SA_temp * self.dt + self.sigA * SA_temp * self.dt * W1
                SB_temp = SB_temp + self.r * SB_temp * self.dt + self.sigB * SB_temp * self.dt * W2
            payoff = payoff + max(0, SB_temp - SA_temp) / self.MC
        return payoff * np.exp(-self.r * self.T)