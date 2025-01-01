import numpy as np

from Seller import Seller


class Consumer:
    def __init__(self, noise_variance):
        self.seller = None
        self.noise_variance = noise_variance
        self.seller_price: float = 0.0

    def switch_seller(self, new_seller: Seller):
        if self.seller:
            noised_price = new_seller.price + np.random.normal(0, self.noise_variance)
            if noised_price < self.seller_price:
                self.seller.consumers.remove(self)
                new_seller.consumers.append(self)
                self.seller = new_seller
        else:
            new_seller.consumers.append(self)
            self.seller = new_seller
        self.seller_price = self.seller.price
