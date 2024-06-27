import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return np.mean(- np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_prime(y_true, y_pred,clip_value=1e-8):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    gradients = y_pred - y_true
    return gradients


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))