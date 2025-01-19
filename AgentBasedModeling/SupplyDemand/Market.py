import logging
import sys

import numpy as np

from Consumer import Consumer
from Seller import Seller


class Market:
    def __init__(self):
        self.sellers: list[Seller] = []
        self.consumers: list[Consumer] = []
        self.sim_data: list[list[dict]] = []
        logging.basicConfig(level=logging.INFO)
        logging.info("Market initialized")

    def set_parameters(self, num_sellers: int, num_consumers: int, seller_epsilon: float, consumer_epsilon: float) -> \
            None:
        self.num_sellers = num_sellers
        self.num_consumers = num_consumers
        self.seller_epsilon = seller_epsilon
        self.consumer_epsilon = consumer_epsilon
        logging.info(
            f"Parameters set: {num_consumers=}, {num_sellers=}, {seller_epsilon=}, {consumer_epsilon=}")

    def compile(self, num_steps: int, num_simulation: int):
        self.num_steps = num_steps
        self.num_simulation = num_simulation
        logging.info(f"Market compiled with {num_steps=} and {num_simulation=}")

    def setup_market(self):
        for seller in range(self.num_sellers):
            self.sellers.append(Seller(seller, np.random.uniform(1, 10), np.random.uniform(0, 5), self.seller_epsilon))
        for consumer in range(self.num_consumers):
            self.consumers.append(
                Consumer(consumer, np.random.uniform(5, 15), np.random.uniform(10, 20), self.consumer_epsilon))
        self.max_price = np.max(
            [consumer.max_price for consumer in self.consumers])
        self.min_price = np.min(
            [seller.min_price for seller in self.sellers])

    def step(self):
        np.random.shuffle(self.consumers)
        for consumer in self.consumers:
            transaction_made = False
            for seller in self.sellers:
                if seller.inventory > 0:
                    if consumer.expected_price >= seller.expected_price:
                        transaction_made = True
                        seller.inventory -= 1
                        consumer.update_expected_price(transaction_made, seller.expected_price)
                        break
            if not transaction_made:
                consumer.update_expected_price(transaction_made, None)

        for seller in self.sellers:
            if seller.inventory > 0:
                seller.update_expected_price(False, None)
            else:
                seller.update_expected_price(True, seller.expected_price)

    def _reset_inventory(self):
        for seller in self.sellers:
            seller.inventory = 1

    def save_data(self, step) -> dict:
        consumer_surplus = sum([consumer.consumer_surplus for consumer in self.consumers if
                                consumer.consumer_surplus != 0])
        seller_surplus = sum([seller.seller_surplus for seller in self.sellers if seller.seller_surplus != 0])
        av_transaction_price = np.mean([seller.transaction_price for seller in self.sellers if
                                        seller.transaction_price is not None])
        consumer_price = np.mean(
            [consumer.expected_price for consumer in self.consumers])

        return {
            'step': step,
            'consumer_surplus': consumer_surplus,
            'seller_surplus': seller_surplus,
            'transaction_price': av_transaction_price,
            'consumer_price': consumer_price,
            'max_price': self.max_price,
            'min_price': self.min_price,
            'consumers': self.consumers,
            'sellers': self.sellers
        }

    def run_simulation(self):
        for sim in range(self.num_simulation):
            self.setup_market()
            step_data = []
            for step in range(self.num_steps):
                self.step()
                step_data.append(self.save_data(step))
                self._reset_inventory()
            self.sim_data.append(step_data)
            self._progress_bar(sim + 1, self.num_simulation)
        logging.info("Simulation finished successfully")

    def _progress_bar(self, current, total, bar_length=40):
        progress = current / total
        block = int(bar_length * progress)
        bar = '#' * block + '-' * (bar_length - block)
        sys.stdout.write(f'\r[{bar}] {current}/{total} ({progress * 100:.2f}%)')
        sys.stdout.flush()

    @staticmethod
    def visualize_simulation(simulation_data: list[dict]):
        pass
