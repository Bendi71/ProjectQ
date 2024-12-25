import numpy as np

from Options.Binomial_option_pricing.crr import CRR


class AmericanOption(CRR):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps):
        super().__init__(asset_price, risk_free_rate, volatility, time_to_maturity, steps)
        self.St = None
        self.V = None

    def _calculate_option_price(self, payoff):
        self.St = self._build_tree()
        self.V = np.zeros_like(self.St)
        self.V[:, -1] = payoff(self.St[:, -1])
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                self.V[j, i] = max(payoff(self.St[j, i]),
                                   self.discount * (self.q * self.V[j, i + 1] + (1 - self.q) * self.V[j + 1, i + 1]))
        return self.V

    def call(self, strike):
        return self._calculate_option_price(np.vectorize(lambda x: max(0, x - strike)))

    def put(self, strike):
        return self._calculate_option_price(np.vectorize(lambda x: max(0, strike - x)))
