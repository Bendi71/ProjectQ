import numpy as np
from numpy import ndarray


class Market:
    def __init__(self):
        self.Total_Market = 0
        self.num_sellers = 0
        self.sim_data = []
        self.history = {"prices": [], "market_shares": []}

    def set_parameters(self, num_consumers: int, num_sellers: int, noise_variance: float, target_market_share: float,
                       epsilon: float,
                       initial_prices: list or float) -> None:
        self.num_consumers = num_consumers
        self.num_sellers = num_sellers
        self.noise_variance = noise_variance
        self.target_market_share = target_market_share
        self.epsilon = epsilon
        self.prices = np.array([initial_prices for _ in range(num_sellers)], dtype=float) \
            if type(initial_prices) == float else np.array(initial_prices, dtype=float)
        self.market_shares = np.zeros(num_sellers)

    def compile(self, steps: int, num_simulations: int) -> None:
        self.steps = steps
        self.num_simulations = num_simulations

    def _assign_consumers_to_sellers(self) -> ndarray:
        proportions = np.linspace(0.1, 1, self.num_sellers)
        proportions /= proportions.sum()
        return np.random.choice(self.num_sellers, size=self.num_consumers, p=proportions)

    def action_supply(self) -> None:
        current_market_shares = np.bincount(self.consumer_sellers, minlength=self.num_sellers) / self.num_consumers
        price_change = np.random.uniform(0, 1, self.num_sellers)
        price_adjustments = np.where(current_market_shares < self.target_market_share,
                                     -self.epsilon * price_change, self.epsilon * price_change)
        self.prices += price_adjustments
        self.prices = np.maximum(self.prices, 0)
        self.market_shares = current_market_shares

    def action_demand(self):
        observed_prices = self.prices[self.consumer_sellers]
        new_sellers = np.random.randint(0, self.num_sellers, self.num_consumers)
        noised_prices = self.prices[new_sellers] + np.random.normal(0, self.noise_variance, self.num_consumers)

        switch_decisions = noised_prices < observed_prices
        self.consumer_sellers = np.where(switch_decisions, new_sellers, self.consumer_sellers)

    def run_simulation(self):
        for sim in range(self.num_simulations):
            self.consumer_sellers = self._assign_consumers_to_sellers()
            step_data = []
            for step in range(self.steps):
                step_data.append(self.save_data(step))
                self.action_supply()
                self.action_demand()
            self.sim_data.append(step_data)

    def save_data(self, step) -> dict:
        avg_price = np.mean(self.prices)
        price_variance = np.var(self.prices)
        num_stay = np.sum(
            np.bincount(self.consumer_sellers, minlength=self.num_sellers) == self.market_shares * self.num_consumers)
        num_switch = self.num_consumers - num_stay
        return {
            "Average Price": avg_price,
            "Price Variance": price_variance,
            "Number Staying": num_stay,
            "Number Switching": num_switch,
            "Prices": self.prices.copy(),
            "Market Shares": self.market_shares.copy(),
            "Step": step
        }
