import numpy as np
from crr import CRR


class AsianOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.in_out = None
        self.barrier = None
        self.barrier_type = None
        self.St = None

    def calculate_option_price(self, payoff):
        self.St = self._build_tree_big() if self.N > 20 else self._build_tree_fast()
        avg_St = np.mean(self.St, axis=1)
        print(avg_St)
        V = payoff(avg_St)
        q_prob = self._calculate_probability()
        return np.sum(V * q_prob) * self.discount ** self.N

    def call(self, strike):
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def put(self, strike):
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))
