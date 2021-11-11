import numpy as np


def relu(X):
    return np.where(X > 0, X, 0)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    nominator = np.exp(X.T)
    return (nominator / np.sum(nominator, axis=1).reshape(len(nominator), 1)).T


def tanh(X):
    return 2 / (1 + np.exp(-2 * X)) - 1


def relu_prime(X):
    return np.where(X > 0, 1, 0)


def sigmoid_prime(X):
    sig = sigmoid(X)
    return sig * (1 - sig)


def tanh_prime(X):
    return 1 - tanh(X)**2
