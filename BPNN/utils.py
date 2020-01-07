import numpy as np
from scipy.special import logsumexp


def sigmoid(data):
    return np.exp(0 - np.logaddexp(0, -data))
    # return 1.0 / (1.0 + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def sigmoid_backward_simple(fx):
    return fx * (1 - fx)


def sigmoid_backward(data):
    fx = sigmoid(data)
    return fx * (1 - fx)


def softmax(data, dim=None):
    # data -> (0, 1, 2)
    # exps = np.exp(data - np.max(data))
    # if dim is not None:
    #     return exps / np.expand_dims(np.sum(exps, axis=dim), dim)
    # else:
    #     return exps / np.sum(exps)

    lse = logsumexp(data, axis=dim)  # (0, 1)
    if dim is not None:
        return np.exp(data - np.expand_dims(lse, dim))
    else:
        return np.exp(data - lse)
