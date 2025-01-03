import numpy as np

from crr import CRR


class LookbackOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.St = None

    def calculate_option_price(self, payoff):
        self.St = self._build_tree_big() if self.N > 20 else self._build_tree_fast()

        if self.N > 20:
            min_St = []
            max_St = []
            for route in self.St:
                min_St.append(min(route))
                max_St.append(max(route))
        else:
            min_St = np.min(self.St, axis=1)
            max_St = np.max(self.St, axis=1)
        V = payoff(min_St, max_St)

        q_prob = self._calculate_probability()

        return np.sum(V * q_prob) * self.discount ** self.N

    def call(self):
        return self.calculate_option_price(np.vectorize(lambda min_st, max_st: np.maximum(0, max_st - self.S0)))

    def put(self):
        return self.calculate_option_price(np.vectorize(lambda min_st, max_st: np.maximum(0, self.S0 - min_st)))


asset_price = 100
risk_free_rate = 0.05
volatility = 0.2
time_to_maturity = 1
steps = 21
strike_price = 100

lookback = LookbackOption(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
print(lookback.call())
