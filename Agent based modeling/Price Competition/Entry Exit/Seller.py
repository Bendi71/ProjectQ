import numpy as np


class Seller:
    def __init__(self, id, initial_price):
        self.id = id
        self.market_share = 0.0
        self.price = initial_price
        self.History = {'MarketShare': [self.market_share], 'Price': [self.price]}
        self.entry_step = 0
        self.exit_step = None
        self.consumers = []

    def price_adjustment(self, target_market_share, epsilon):
        price_change = epsilon * np.random.rand()
        if self.market_share < target_market_share:
            self.price = np.maximum(0, self.price - price_change)
        else:
            self.price = self.price + price_change

    def add_consumers(self, consumers: list[object]):
        self.consumers.extend(consumers)

    def calculate_market_share(self, num_consumers: int):
        return len(self.consumers) / num_consumers

    def save_step_data(self):
        self.History['MarketShare'].append(self.market_share)
        self.History['Price'].append(self.price)
