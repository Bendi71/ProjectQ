import numpy as np

from crr import CRR


class ChooserOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps, chooser_time):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.St = None
        self.chooser_time = chooser_time
        self.chooser_step = int(chooser_time * steps / time_to_maturity)

    def _calculate_option_price(self, payoff_call, payoff_put):
        self.St = self._build_tree_big() if self.N > 20 else self._build_tree_fast()
        if self.N > 20:
            call_payoff = []
            put_payoff = []
            for route in self.St:
                call_payoff.append(payoff_call(route[self.chooser_step]))
                put_payoff.append(payoff_put(route[:, self.chooser_step]))
        else:
            call_payoff = payoff_call(self.St[:, self.chooser_step])
            put_payoff = payoff_put(self.St[:, self.chooser_step])
        chooser_payoff = np.maximum(call_payoff, put_payoff)

        q_prob = self._calculate_probability()

        return np.sum(chooser_payoff * q_prob) * self.discount ** self.N

    def chooser(self, strike):
        return self._calculate_option_price(
            np.vectorize(lambda x: np.maximum(0, x - strike)),
            np.vectorize(lambda x: np.maximum(0, strike - x))
        )