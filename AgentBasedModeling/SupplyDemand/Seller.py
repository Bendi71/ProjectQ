class Seller:
    def __init__(self, id, initial_price, min_price, epsilon):
        self.id = id
        self.min_price = min_price
        self.expected_price = initial_price
        self.transaction_price = None
        self.epsilon = epsilon
        self.inventory = 1
        self.seller_surplus = 0

    def update_expected_price(self, transaction_made, transaction_price):
        if transaction_made:
            self.transaction_price = self.expected_price
            self.expected_price = max(self.min_price, self.expected_price + self.epsilon * self.expected_price)
            self.calculate_surplus(transaction_price)
        else:
            self.transaction_price = None
            self.expected_price = max(self.min_price, self.expected_price - self.epsilon *
                                      self.expected_price)
            self.seller_surplus = 0

    def calculate_surplus(self, transaction_price):
        self.seller_surplus = transaction_price - self.min_price
