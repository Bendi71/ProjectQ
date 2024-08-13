from crr import CRR
import numpy as np


class BarrierOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.barrier = None
        self.St = None
        self.in_out = None
        self.barrier_type = None

    def calculate_option_price(self, payoff):
        self.St = self._build_tree_big() if self.N > 20 else self._build_tree_fast()

        max_St = np.max(self.St, axis=1)
        min_St = np.min(self.St, axis=1)
        last_St = self.St[:, -1]

        if self.barrier_type == 'up':
            if self.in_out == 'out':
                valid_indices = max_St <= self.barrier
            else:
                valid_indices = max_St >= self.barrier
        else:
            if self.in_out == 'out':
                valid_indices = min_St >= self.barrier
            else:
                valid_indices = min_St <= self.barrier

        last_St = np.where(valid_indices, last_St, 0)
        V = payoff(last_St)

        q_prob = self._calculate_probability()
        return np.sum(V * q_prob) * self.discount ** self.N

    def call(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'up'
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def put(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'down'
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))

    def up_and_out_call(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'up'
        self.in_out = 'out'
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def up_and_out_put(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'up'
        self.in_out = 'out'
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))

    def down_and_out_call(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'down'
        self.in_out = 'out'
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def down_and_out_put(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'down'
        self.in_out = 'out'
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))

    def up_and_in_call(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'down'
        self.in_out = 'in'
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def up_and_in_put(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'down'
        self.in_out = 'in'
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))

    def down_and_in_call(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'up'
        self.in_out = 'in'
        return self.calculate_option_price(lambda x: np.maximum(0, x - strike))

    def down_and_in_put(self, strike, barrier):
        self.barrier = barrier
        self.barrier_type = 'up'
        self.in_out = 'in'
        return self.calculate_option_price(lambda x: np.maximum(0, strike - x))
