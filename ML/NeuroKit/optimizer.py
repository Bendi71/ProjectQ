import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradient):
        return weights - self.learning_rate * gradient


class Adam:
    # FIXME: Implement Adam optimizer
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def update(self, params, grads):
        if self.m is None or self.v is None:
            self.initialize(params.shape)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        updated_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_params
