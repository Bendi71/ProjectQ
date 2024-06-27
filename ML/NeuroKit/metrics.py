import numpy as np


def accuracy(y_true, y_pred, threshold=0.5):
    if y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        if y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
    else:
        y_pred = y_pred > threshold
    return np.mean(np.equal(y_true, y_pred))

def r2_score(y_true, y_pred):
    return 1 - np.sum(np.power(y_true - y_pred, 2)) / np.sum(np.power(y_true - np.mean(y_true), 2))