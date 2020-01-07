# -*- coding: utf-8 -*-

import pickle

import numpy as np


class HMM(object):

    def __init__(self, nw, nt):
        # 词汇数量
        self.nw = nw
        # 词性数量
        self.nt = nt

    def train(self, trainset, alpha, file):
        A = np.zeros((self.nt + 1, self.nt + 1))
        B = np.zeros((self.nw, self.nt))

        for wiseq, tiseq in trainset:
            prev = -1
            for wi, ti in zip(wiseq, tiseq):
                A[ti, prev] += 1
                B[wi, ti] += 1
                prev = ti
            A[self.nt, prev] += 1
        A = self.smooth(A, alpha)

        # 迁移概率
        self.trans = np.log(A[:-1, :-1])
        # 句首迁移概率
        self.strans = np.log(A[:-1, -1])
        # 句尾迁移概率
        self.etrans = np.log(A[-1, :-1])
        # 发射概率
        self.emit = np.log(self.smooth(B, alpha))

        # 保存训练好的模型
        if file is not None:
            self.dump(file)

    def smooth(self, matrix, alpha):
        sums = np.sum(matrix, axis=0)
        return (matrix + alpha) / (sums + alpha * len(matrix))

    def predict(self, wiseq):
        T, N = len(wiseq), self.nt
        delta = np.zeros((T, N))
        paths = np.zeros((T, N), dtype='int')

        delta[0] = self.strans + self.emit[wiseq[0]]

        for i in range(1, T):
            probs = self.trans + delta[i - 1]
            paths[i] = np.argmax(probs, axis=1)
            delta[i] = probs[np.arange(N), paths[i]] + self.emit[wiseq[i]]
        prev = np.argmax(delta[-1] + self.etrans)

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return predict

    def evaluate(self, data):
        tp, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            predict = np.array(self.predict(wiseq))
            tp += np.sum(tiseq == predict)
        accuracy = tp / total
        return tp, total, accuracy

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            hmm = pickle.load(f)
        return hmm
