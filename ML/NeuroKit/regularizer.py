import numpy as np

class Regularizer:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def penalty(self, weights):
        raise NotImplementedError

    def gradient(self, weights):
        raise NotImplementedError

class L1(Regularizer):
    def penalty(self, weights):
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.lambda_ * np.sign(weights)

class L2(Regularizer):
    def penalty(self, weights):
        return self.lambda_ * np.sum(weights**2) * 0.5

    def gradient(self, weights):
        return self.lambda_ * weights

class L1L2(Regularizer):
    def __init__(self, lambda_l1, lambda_l2):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def penalty(self, weights):
        return self.lambda_l1 * np.sum(np.abs(weights)) + self.lambda_l2 * np.sum(weights**2) * 0.5

    def gradient(self, weights):
        return self.lambda_l1 * np.sign(weights) + self.lambda_l2 * weights


