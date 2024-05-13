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
