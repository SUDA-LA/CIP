from scipy.special import logsumexp
from .module import Module
import numpy as np


class LinearLayer(Module):
    def __init__(self, input_dim, output_dim, random_init):
        super(LinearLayer, self).__init__()
        if random_init:
            self.weight = np.random.randn(input_dim, output_dim) / np.sqrt(output_dim)
            self.bias = np.random.randn(output_dim)
        else:
            self.weight = np.zeros((input_dim, output_dim))
            self.bias = np.zeros(output_dim)

    def forward(self, x):
        self.input = x
        y = x @ self.weight  # (batch, hidden)
        y += self.bias
        self.output = y
        return y

    def backward(self, grad, lr, normal):
        assert self.output is not None and self.input is not None
        # grad -> (batch, output)
        weight_grad = self.input.T @ grad  # (input, batch) @ (batch, output) -> (input, output)
        out_grad = grad @ self.weight.T  # (batch, output) @ (output, input) -> (batch, input)
        self.weight = normal * self.weight - lr * weight_grad
        self.bias -= lr * np.sum(grad, 0)
        return out_grad


class EmbeddingLayer(Module):
    def __init__(self, num_embeddings, embedding_dim, random_init):
        super(EmbeddingLayer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if random_init:
            self.embed = np.random.randn(num_embeddings, embedding_dim) / np.sqrt(embedding_dim)
        else:
            self.embed = np.zeros((num_embeddings, embedding_dim))

    def forward(self, x):
        self.input = x
        data = self.embed[x]  # (batch_size, input_size, embedding_size)
        self.output = data

    def backward(self, grad, lr):
        using_data = list(set(self.input.reshape(-1)))

        for data in using_data:
            self.embed[data] -= lr * np.sum(grad[self.input == data], 0)
        pass


class SigmoidLayer(Module):
    def __init__(self):
        super(SigmoidLayer, self).__init__()

    def forward(self, x):
        self.input = x
        y = np.exp(0 - np.logaddexp(0, -x))
        self.output = y
        return y

    def backward(self, grad):
        assert self.output is not None and self.input is not None
        fx = self.output
        return grad * fx * (1 - fx)


class ReluLayer(Module):
    def __init__(self):
        super(ReluLayer, self).__init__()

    def forward(self, x):
        self.input = x
        y = np.maximum(0, x)
        self.output = y
        return y

    def backward(self, grad):
        assert self.output is not None and self.input is not None
        fx = np.float32(self.output > 0)
        return fx


class SoftmaxLayer(Module):
    def __init__(self):
        super(SoftmaxLayer, self).__init__()

    def forward(self, x, dim=None):
        self.input = x
        lse = logsumexp(x, axis=dim)  # (0, 1)
        if dim is not None:
            y = np.exp(x - np.expand_dims(lse, dim))
        else:
            y = np.exp(x - lse)
        self.output = y
        return y

    def backward(self, y, loss_type='cross_entropy_loss'):
        assert self.output is not None and self.input is not None
        if loss_type == 'cross_entropy_loss':
            return self.output - y
        else:
            # TBD
            return self.output - y
