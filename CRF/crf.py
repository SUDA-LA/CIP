# -*- coding: utf-8 -*-

import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.misc import logsumexp


class CRF(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wiseq, tiseq in data
            for f in set(self.instantiate(wiseq, 0, -1, tiseq[0])).union(
                *[self.instantiate(wiseq, i, tiseq[i - 1], ti)
                  for i, ti in enumerate(tiseq[1:], 1)]
            )
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.d)
        # Bigram特征及对应权重分值
        self.BF = [
            [self.bigram(prev_ti, ti) for prev_ti in range(self.nt)]
            for ti in range(self.nt)
        ]
        self.BS = np.zeros((self.nt, self.nt))

    def SGD(self, trainset, devset, file,
            epochs, batch_size, interval, eta, decay, lmbda,
            anneal, regularize):
        # 训练集大小
        n = len(trainset)
        # 记录更新次数
        count = 0
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0

        # 迭代指定次数训练模型
        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 随机打乱数据
            random.shuffle(trainset)
            # 设置L2正则化系数
            if not regularize:
                lmbda = 0
            # 按照指定大小对数据分割批次
            batches = [trainset[i:i + batch_size]
                       for i in range(0, len(trainset), batch_size)]
            nb = len(batches)
            # 根据批次数据更新权重
            for batch in batches:
                if not anneal:
                    self.update(batch, lmbda, n, eta)
                # 设置学习速率的指数衰减
                else:
                    self.update(batch, lmbda, n, eta * decay ** (count / nb))
                count += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            tp, total, accuracy = self.evaluate(trainset)
            print("%-6s %d / %d = %4f" % ('train:', tp, total, accuracy))
            tp, total, accuracy = self.evaluate(devset)
            print("%-6s %d / %d = %4f" % ('dev:', tp, total, accuracy))
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
        print("mean time of each epoch is %ss\n" % (total_time / epoch))

    def update(self, batch, lmbda, n, eta):
        gradients = defaultdict(float)

        for wiseq, tiseq in batch:
            prev_ti = -1
            for i, ti in enumerate(tiseq):
                fiseq = (self.fdict[f]
                         for f in self.instantiate(wiseq, i, prev_ti, ti)
                         if f in self.fdict)
                for fi in fiseq:
                    gradients[fi] += 1
                prev_ti = ti

            alpha = self.forward(wiseq)
            beta = self.backward(wiseq)
            logZ = logsumexp(alpha[-1])

            for i, ti in enumerate(range(self.nt)):
                fv = self.instantiate(wiseq, 0, -1, ti)
                fiseq = (self.fdict[f] for f in fv if f in self.fdict)
                p = np.exp(self.score(fv) + beta[0, i] - logZ)
                for fi in fiseq:
                    gradients[fi] -= p

            for i in range(1, len(tiseq)):
                for j, ti in enumerate(range(self.nt)):
                    ufv = self.unigram(wiseq, i, ti)
                    ufiseq = [self.fdict[f] for f in ufv if f in self.fdict]
                    scores = self.BS[j] + self.score(ufv)
                    probs = np.exp(scores + alpha[i - 1] + beta[i, j] - logZ)

                    for bfv, p in zip(self.BF[j], probs):
                        bfiseq = [self.fdict[f]
                                  for f in bfv if f in self.fdict]
                        for fi in bfiseq + ufiseq:
                            gradients[fi] -= p

        if lmbda != 0:
            self.W *= (1 - eta * lmbda / n)
        for k, v in gradients.items():
            self.W[k] += eta * v
        self.BS = np.array([
            [self.score(bfv) for bfv in bfvs]
            for bfvs in self.BF
        ])

    def forward(self, wiseq):
        T = len(wiseq)
        alpha = np.zeros((T, self.nt))

        fvs = [self.instantiate(wiseq, 0, -1, ti)
               for ti in range(self.nt)]
        alpha[0] = [self.score(fv) for fv in fvs]

        for i in range(1, T):
            uscores = np.array([
                self.score(self.unigram(wiseq, i, ti))
                for ti in range(self.nt)
            ]).reshape(-1, 1)
            scores = self.BS + uscores
            alpha[i] = logsumexp(alpha[i - 1] + scores, axis=1)
        return alpha

    def backward(self, wiseq):
        T = len(wiseq)
        beta = np.zeros((T, self.nt))

        for i in reversed(range(T - 1)):
            uscores = np.array([
                self.score(self.unigram(wiseq, i + 1, ti))
                for ti in range(self.nt)
            ])
            scores = self.BS.T + uscores
            beta[i] = logsumexp(beta[i + 1] + scores, axis=1)
        return beta

    def predict(self, wiseq):
        T = len(wiseq)
        delta = np.zeros((T, self.nt))
        paths = np.zeros((T, self.nt), dtype='int')

        fvs = [self.instantiate(wiseq, 0, -1, ti)
               for ti in range(self.nt)]
        delta[0] = [self.score(fv) for fv in fvs]

        for i in range(1, T):
            uscores = np.array([
                self.score(self.unigram(wiseq, i, ti))
                for ti in range(self.nt)
            ]).reshape(-1, 1)
            scores = self.BS + uscores + delta[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            delta[i] = scores[np.arange(self.nt), paths[i]]
        prev = np.argmax(delta[-1])

        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()
        return predict

    def score(self, fvector):
        scores = [self.W[self.fdict[f]]
                  for f in fvector if f in self.fdict]
        return sum(scores)

    def bigram(self, prev_ti, ti):
        return [('01', ti, prev_ti)]

    def unigram(self, wiseq, index, ti):
        word = wiseq[index]
        prev_word = wiseq[index - 1] if index > 0 else '^^'
        next_word = wiseq[index + 1] if index < len(wiseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', ti, word))
        fvector.append(('03', ti, prev_word))
        fvector.append(('04', ti, next_word))
        fvector.append(('05', ti, word, prev_char))
        fvector.append(('06', ti, word, next_char))
        fvector.append(('07', ti, first_char))
        fvector.append(('08', ti, last_char))

        for char in word[1: -1]:
            fvector.append(('09', ti, char))
            fvector.append(('10', ti, first_char, char))
            fvector.append(('11', ti, last_char, char))
        if len(word) == 1:
            fvector.append(('12', ti, word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', ti, char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', ti, word[: i]))
                fvector.append(('15', ti, word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', ti, word))
            fvector.append(('15', ti, word))
        return fvector

    def instantiate(self, wiseq, index, prev_ti, ti):
        bigram = self.bigram(prev_ti, ti)
        unigram = self.unigram(wiseq, index, ti)
        return bigram + unigram

    def evaluate(self, data):
        tp, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            piseq = np.array(self.predict(wiseq))
            tp += np.sum(tiseq == piseq)
        accuracy = tp / total
        return tp, total, accuracy

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            crf = pickle.load(f)
        return crf
