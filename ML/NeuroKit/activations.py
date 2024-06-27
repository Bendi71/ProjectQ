import numpy as np
from .activation import Activation


class Relu(Activation):
    def __init__(self):
        super().__init__(self.relu, self.relu_prime)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return 1 * (x > 0)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def softmax_prime(self, x):
        return 1


class Linear(Activation):
    def __init__(self):
        super().__init__(self.linear, self.linear_prime)

    def linear(self, x):
        return x

    def linear_prime(self, x):
        return 1
