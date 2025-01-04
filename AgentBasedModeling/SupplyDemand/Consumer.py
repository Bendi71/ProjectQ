class Consumer:
    def __init__(self, id, max_price, expected_price, epsilon):
        self.id = id
        self.max_price = max_price
        self.expected_price = expected_price
        self.epsilon = epsilon
        self.consumer_surplus = 0

    def update_expected_price(self, transaction_made, transaction_price):
        if transaction_made:
            self.expected_price = min(self.max_price, self.expected_price - self.epsilon * self.expected_price)
            self.calculate_surplus(transaction_price)
        else:
            self.expected_price = min(self.max_price, self.expected_price + self.epsilon * self.expected_price)
            self.consumer_surplus = 0

    def calculate_surplus(self, transaction_price):
        self.consumer_surplus = self.max_price - transaction_price
