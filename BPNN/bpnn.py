# -*- coding: utf-8 -*-

import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from utils import sigmoid, sigmoid_prime, softmax


class BPNN(object):

    def __init__(self, vocdim, window, sizes, embed):
        # 上下文窗口大小
        self.window = window
        # 词嵌入向量维度
        self.embdim = sizes[0] // window
        # 神经网络每层神经元的数量
        self.sizes = sizes
        # 神经网络层数
        self.nl = len(sizes)
        # 词性数量
        self.nt = sizes[-1]

        # 词嵌入矩阵
        self.embed = embed
        # 神经网络偏置
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 神经网络权重
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def SGD(self, trainset, devset, file,
            epochs, batch_size, interval, eta, lmbda, epsilon=1e-8,
            adagrad=False):
        n = len(trainset)
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0

        if adagrad:
            self.GB = [np.zeros(b.shape) + epsilon for b in self.biases]
            self.GW = [np.zeros(w.shape) + epsilon for w in self.weights]
        for epoch in range(epochs):
            start = datetime.now()

            random.shuffle(trainset)
            batches = [trainset[k:k + batch_size]
                       for k in range(0, n, batch_size)]

            for batch in batches:
                self.update(batch, eta, lmbda, n, adagrad=adagrad)

            print("Epoch %d / %d: " % (epoch, epochs))
            loss = self.loss(trainset, lmbda)
            tp, total, accuracy = self.evaluate(trainset)
            print("%-6s Loss: %4f Accuracy: %d / %d = %4f" %
                  ('train:', loss, tp, total, accuracy))
            loss = self.loss(devset, lmbda)
            tp, total, accuracy = self.evaluate(devset)
            print("%-6s Loss: %4f Accuracy: %d / %d = %4f" %
                  ('dev:', loss, tp, total, accuracy))
            t = datetime.now() - start
            print("%ss elapsed\n" % t)
            total_time += t

            # 保存效果最好的模型
            if accuracy > max_accuracy:
                self.dump(file)
                max_e, max_accuracy = epoch, accuracy
            elif epoch - max_e > interval:
                break
        print("max accuracy of dev is %4f at epoch %d" %
              (max_accuracy, max_e))
        print("mean time of each epoch is %ss\n" % (total_time / (epoch + 1)))

    def update(self, batch, eta, lmbda, n, adagrad=False):
        n_batch = len(batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_x = defaultdict(float)

        for x, y in batch:
            dnabla_b, dnabla_w, dnabla_x = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, dnabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dnabla_w)]
            for wi, dnx in zip(x, np.reshape(dnabla_x, (-1, self.embdim))):
                nabla_x[wi] += dnx
        if adagrad:
            # 累加梯度平方
            self.GB = [gb + nb ** 2 for gb, nb in zip(self.GB, nabla_b)]
            self.GW = [gw + nw ** 2 for gw, nw in zip(self.GW, nabla_w)]
            # 根据累加梯度更新当前梯度
            nabla_b = [nb / np.sqrt(gb) for gb, nb in zip(self.GB, nabla_b)]
            nabla_w = [nw / np.sqrt(gw) for gw, nw in zip(self.GW, nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / n_batch) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / n_batch) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        for k, nx in nabla_x.items():
            self.embed[k] -= (eta / n_batch) * nx

    def forward(self, x, getall=False):
        # 将输入经过词嵌入矩阵转换成拼接的词向量
        a = np.reshape(self.embed[x], (-1, 1))
        zs, activations = [], [a]
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        zs.append(z)
        # 输出层利用softmax计算激活值
        a = softmax(z)
        activations.append(a)
        return (zs, activations) if getall else a

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播
        zs, activations = self.forward(x, getall=True)

        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.nl):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        nabla_x = np.dot(self.weights[0].T, delta)
        return nabla_b, nabla_w, nabla_x

    def evaluate(self, data):
        total = len(data)
        tp = sum([
            y[np.argmax(self.forward(x))] for (x, y) in data
        ])
        return tp, total, tp / total

    def loss(self, data, lmbda):
        loss = 0.0
        for x, y in data:
            a = self.forward(x)
            loss -= np.log(a[np.argmax(y)])
        # 计算正则项
        loss += 0.5 * lmbda * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        loss /= len(data)
        return loss

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            bpnn = pickle.load(f)
        return bpnn
