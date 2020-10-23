# 工具类
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_backward(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(data):
    return np.maximum(data, 0)

def relu_backward(z):
    return np.int64(z > 0)

def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)
