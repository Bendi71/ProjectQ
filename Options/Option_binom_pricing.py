"""
Cox-Ross-Rubinstein (CRR) Option Pricing Model Project

This project aims to implement the Cox-Ross-Rubinstein (CRR) model for option pricing. The CRR model is a binomial tree model that is used to calculate the theoretical price of options. It was developed by John Cox, Stephen Ross, and Mark Rubinstein in 1979.

The CRR model works by dividing the time to expiration of the option into a number of time intervals, or steps. At each step, it is assumed that the price of the underlying asset can move up or down by a certain factor. This leads to a binomial tree of asset prices. The option prices at each step are then calculated backwards from the expiration date to the present, using the concept of risk-neutral valuation.

The up and down factors in the CRR model are calculated using the following equations:

    u = exp(sigma * sqrt(delta_t))
    d = 1 / u

where:
    u = up factor
    d = down factor
    sigma = volatility of the underlying asset
    delta_t = length of a time step

The risk-neutral probability (q) in the CRR model is calculated using the following equation:

    q = (exp(r * delta_t) - d) / (u - d)

where:
    r = risk-free interest rate

The CRR model is particularly useful because it converges to the Black-Scholes model as the number of steps increases, making it a good model for pricing American options and other exotic options.

The code is written in Python and uses libraries such as numpy for numerical calculations.
"""
import numpy as np

class Barrier_option:
    def __init__(self, asset_price, strike_price, upper_strike, lower_strike, riskfreerate, volatility,
                 time_toexpiariton, steps):
        self.S = asset_price
        self.K = strike_price
        self.Kup = upper_strike
        self.Kdo = lower_strike
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.up_prob = np.exp(self.sig * np.sqrt(self.dt))
        self.do_prob = 1 / self.up_prob
        self.q_prob = (np.exp(self.r * self.dt) - self.do_prob) / (self.up_prob - self.do_prob)

    def upout_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        g[ST >= self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST >= self.Kup] = 0
        return g

    def upout_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        g[ST >= self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST >= self.Kup] = 0
        return g

    def downout_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        g[ST <= self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST <= self.Kdo] = 0
        return g

    def downout_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        g[ST <= self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST <= self.Kdo] = 0
        return g

    def upin_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        C = g.copy()
        g[ST < self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] >= self.Kup:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def upin_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        C = g.copy()
        g[ST < self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] >= self.Kup:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def downin_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        C = g.copy()
        g[ST > self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] <= self.Kdo:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def downin_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        C = g.copy()
        g[ST > self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] <= self.Kdo:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]


class American_option:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, steps):
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.up_prob = np.exp(self.sig * np.sqrt(self.dt))
        self.do_prob = 1 / self.up_prob
        self.q_prob = (np.exp(self.r * self.dt) - self.do_prob) / (self.up_prob - self.do_prob)

    def am_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        C = g.copy()
        for i in np.arange(self.N - 1, -1, -1):
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            intrin = np.maximum(ST - self.K, 0)
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            for j in range(0, i + 1):
                if intrin[j] > C[j]:
                    g[j] = intrin[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def am_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        C = g.copy()
        for i in np.arange(self.N - 1, -1, -1):
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            intrin = np.maximum(self.K - ST, 0)
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            for j in range(0, i + 1):
                if intrin[j] > C[j]:
                    g[j] = intrin[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]


class Chooser_option:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, time_tochoose, steps):
        self.TC = time_tochoose
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.up_prob = np.exp(self.sig * np.sqrt(self.dt))
        self.do_prob = 1 / self.up_prob
        self.q_prob = (np.exp(self.r * self.dt) - self.do_prob) / (self.up_prob - self.do_prob)

    def chooser(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        C = np.maximum(ST - self.K, 0)
        P = np.maximum(self.K - ST, 0)
        close_step = min(range(0, self.N), key=lambda x: abs(x * self.dt - self.TC))
        g = np.zeros(close_step + 1)
        for i in np.arange(self.N - 1, close_step - 1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            P[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * P[1:i + 2] + (1 - self.q_prob) * P[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
        for j in range(0, close_step + 1):
            if ST[j] >= self.K:
                g[j] = C[j]
            else:
                g[j] = P[j]
        for i in np.arange(close_step - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
        return g
class ExchangeableOption:
    def __init__(self, asset_price_A, asset_price_B, riskfreerate, volatility_A, volatility_B, correlation, time_to_expiration, steps):
        self.SA = asset_price_A
        self.SB = asset_price_B
        self.r = riskfreerate
        self.sigA = volatility_A
        self.sigB = volatility_B
        self.rho = correlation
        self.T = time_to_expiration
        self.N = steps


    def _update_prices(self, S1, S2, r, sigma1, sigma2, rho, n, eps1, eps2):
        S1_new = S1 * (1 + r / n + sigma1 / np.sqrt(n) * eps1)
        S2_new = S2 * (1 + r / n + sigma2 / np.sqrt(n) * (rho * eps1 + np.sqrt(1 - rho ** 2) * eps2))
        return S1_new, S2_new

    def _build_tree(self):
        import itertools
        letters = ['u', 'm', 'd']
        S1_tree = []
        S2_tree = []

        eps_map = {
            'u': (np.sqrt(3 / 2), 1 / np.sqrt(2)),
            'm': (0, -np.sqrt(2)),
            'd': (-np.sqrt(3 / 2), 1 / np.sqrt(2))
        }

        for combo in itertools.product(letters, repeat=self.N):
            S1_new, S2_new = self.SA, self.SB
            for move in combo:
                eps1, eps2 = eps_map[move]
                S1_new, S2_new = self._update_prices(S1_new, S2_new, self.r, self.sigA, self.sigB, self.rho, self.N, eps1, eps2)
            S1_tree.append(S1_new)
            S2_tree.append(S2_new)

        return np.array(S1_tree), np.array(S2_tree)

    def _calculate_option_price(self, payoff_func):
        S1_tree, S2_tree = self._build_tree()
        payoff = payoff_func(S1_tree, S2_tree)
        probabilities = 1 / 3 ** self.N
        g = np.sum(payoff) * probabilities
        return g * np.exp(-self.r * self.T)

    def call(self):
        return self._calculate_option_price(lambda S1, S2: np.maximum(S1 - S2, 0))

    def put(self):
        return self._calculate_option_price(lambda S1, S2: np.maximum(S2 - S1, 0))