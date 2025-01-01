import logging

import numpy as np

from Consumer import Consumer
from Seller import Seller


class Market:
    def __init__(self) -> None:
        self.sellers: list[Seller] = []
        self.consumers: list[Consumer] = []
        self.av_price = 0
        self.num_sellers = 0
        self.sim_data: list[list[dict]] = []
        self.sim_all_sellers: list[list[Seller]] = []
        logging.info("Market initialized")

    def set_parameters(self, num_consumers: int, num_sellers: int, noise_variance: float, target_market_share: float,
                       epsilon: float, initial_price: float, entry_probability: float = 0.05, exit_threshold: float
                       = 0.10) -> None:
        self.num_consumers = num_consumers
        self.num_sellers = num_sellers
        self.noise_variance = noise_variance
        self.target_market_share = target_market_share
        self.epsilon = epsilon
        self.initial_price = initial_price
        self.market_shares = np.zeros(num_sellers)
        self.entry_threshold = entry_probability
        self.exit_threshold = exit_threshold
        self.consumers = [Consumer(noise_variance) for _ in range(num_consumers)]
        logging.info(
            f"Parameters set: {num_consumers=}, {num_sellers=}, {noise_variance=}, {target_market_share=}, {epsilon=}, {initial_price=}, {entry_probability=}, {exit_threshold=}")

    def compile(self, steps: int, num_simulations: int) -> None:
        self.steps = steps
        self.num_simulations = num_simulations
        logging.info(f"Market compiled with {steps=} and {num_simulations=}")

    def _assign_initial_sellers(self) -> None:
        proportions = np.linspace(0.1, 1, self.num_sellers)
        proportions /= proportions.sum()
        for consumer in self.consumers:
            seller: Seller = np.random.choice(self.sellers, p=proportions)
            seller.add_consumers([consumer])
            consumer.seller = seller
            consumer.seller_price = seller.price

    def step(self) -> None:
        for seller in self.sellers:
            seller.market_share = seller.calculate_market_share(self.num_consumers)
            seller.price_adjustment(self.target_market_share, self.epsilon)
            seller.save_step_data()

        for consumer in self.consumers:
            new_seller: Seller = np.random.choice(self.sellers)
            consumer.switch_seller(new_seller)

    def _new_seller(self, step):
        if np.random.rand() < self.entry_threshold:
            new_seller = Seller(len(self.all_sellers), np.mean([seller.price for seller in self.sellers]))
            new_seller.entry_step = step
            self.sellers.append(new_seller)
            self.all_sellers.append(new_seller)
            logging.info(f"New seller added at step {step}")

    def _remove_seller(self, step):
        remaining_sellers = []
        for seller in self.sellers:
            if seller.market_share > self.exit_threshold:
                remaining_sellers.append(seller)
            else:
                seller.History['MarketShare'].append(seller.market_share)
                seller.History['Price'].append(seller.price)
                seller.exit_step = step
                for consumer in seller.consumers:
                    consumer.seller = None
        self.sellers = remaining_sellers
        logging.info(f"Sellers removed at step {step}")

    def save_data(self, step) -> dict:
        avg_price = np.mean([seller.price for seller in self.sellers])
        price_variance = np.var([seller.price for seller in self.sellers])
        num_stay = np.sum(
            np.bincount([seller.id for seller in self.sellers], minlength=len(self.sellers)) ==
            self.market_shares *
            self.num_consumers)
        num_switch = self.num_consumers - num_stay
        return {
            "avg_price": avg_price,
            "price_variance": price_variance,
            "num_sellers": len(self.sellers),
            "num_switch": num_switch,
            "step": step
        }

    def run_simulation(self):
        for sim in range(self.num_simulations):
            self.sellers = [Seller(id, self.initial_price) for id in range(self.num_sellers)]
            self.all_sellers = self.sellers.copy()
            self._assign_initial_sellers()
            step_data = []
            for step in range(self.steps):
                self.step()
                step_data.append(self.save_data(step))
                self._remove_seller(step)
                self._new_seller(step)
            self.sim_data.append(step_data)
            self.sim_all_sellers.append(self.all_sellers)
            logging.info(f"Simulation {sim + 1}/{self.num_simulations} completed")

    @staticmethod
    def reshape_MS_data(all_sellers: list, steps: int) -> np.ndarray:
        reshaped_data = np.full((len(all_sellers), steps), np.nan)

        for idx, seller in enumerate(all_sellers):
            start_step = seller.entry_step
            end_step = seller.exit_step if seller.exit_step is not None else steps

            for step in range(start_step, end_step):
                history_index = step - start_step
                if history_index < len(seller.History['MarketShare']):
                    reshaped_data[idx, step] = seller.History['MarketShare'][history_index]

        return reshaped_data

    @staticmethod
    def reshape_P_data(all_sellers: list, steps: int) -> np.ndarray:
        reshaped_data = np.full((len(all_sellers), steps), np.nan)

        for idx, seller in enumerate(all_sellers):
            start_step = seller.entry_step
            end_step = seller.exit_step if seller.exit_step is not None else steps

            for step in range(start_step, end_step):
                history_index = step - start_step
                if history_index < len(seller.History['Price']):
                    reshaped_data[idx, step] = seller.History['Price'][history_index]

        return reshaped_data
