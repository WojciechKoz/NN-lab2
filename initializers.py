import numpy as np
from math import sqrt


class StandardInitializer:
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def init_weights(self, neurons):
        w, b = [], []
        for i in range(len(neurons)-1):
            w.append(np.random.normal(self.mean, self.std, size=(neurons[i + 1], neurons[i])))
            b.append(np.random.normal(self.mean, self.std, size=(neurons[i + 1], 1)))
        return w, b


class XavierInitializer:
    def __init__(self, mean=0):
        self.mean = mean

    def init_weights(self, neurons):
        w, b = [], []
        for i in range(len(neurons)-1):
            w.append(np.random.normal(self.mean, sqrt(2.0 / (neurons[i+1] + neurons[i])), size=(neurons[i+1], neurons[i])))
            b.append(np.random.normal(self.mean, sqrt(2.0 / (neurons[i+1] + neurons[i])), size=(neurons[i+1], 1)))
        return w, b


class HeInitializer:
    def __init__(self, mean=0):
        self.mean = mean

    def init_weights(self, neurons):
        w, b = [], []
        for i in range(len(neurons)-1):
            w.append(np.random.normal(self.mean, sqrt(2.0 / (neurons[i])), size=(neurons[i+1], neurons[i])))
            b.append(np.random.normal(self.mean, sqrt(2.0 / (neurons[i])), size=(neurons[i+1], 1)))
        return w, b
