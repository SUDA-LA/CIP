# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)
