from crr import CRR
import numpy as np


class ChooserOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps, chooser_time):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.St = None
        self.chooser_time = chooser_time
        self.chooser_step = int(chooser_time * steps / time_to_maturity)

    def calculate_option_price(self, payoff_call, payoff_put):
        self.St = self._build_tree_big() if self.N > 20 else self._build_tree_fast()

        call_payoff = payoff_call(self.St[:, self.chooser_step])
        put_payoff = payoff_put(self.St[:, self.chooser_step])
        chooser_payoff = np.maximum(call_payoff, put_payoff)

        q_prob = self._calculate_probability()

        return np.sum(chooser_payoff * q_prob) * self.discount ** self.N

    def chooser(self, strike):
        return self.calculate_option_price(
            lambda x: np.maximum(0, x - strike),
            lambda x: np.maximum(0, strike - x)
        )


if __name__ == '__main__':
    option = ChooserOption(100, 0.05, 0.2, 1, 3, 0.5)
    print(option.chooser(100))
