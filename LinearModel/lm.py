# -*- coding: utf-8 -*-

import pickle
import random
from collections import Counter
from datetime import datetime, timedelta

import numpy as np


class LinearModel(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wiseq, tiseq in data
            for i, ti in enumerate(tiseq)
            for f in self.instantiate(wiseq, i, ti)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.d)
        # 累加特征权重
        self.V = np.zeros(self.d)

    def online(self, trainset, devset, file, epochs, interval, average):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_accuracy = 0, 0.0

        # 迭代指定次数训练模型
        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 随机打乱数据
            random.shuffle(trainset)
            # 保存更新时间戳和每个特征最近更新时间戳的记录
            self.k, self.R = 0, np.zeros(self.d, dtype='int')
            for batch in trainset:
                self.update(batch)
            self.V += [(self.k - r) * w for r, w in zip(self.R, self.W)]

            print("Epoch %d / %d: " % (epoch, epochs))
            tp, total, accuracy = self.evaluate(trainset, average=average)
            print("%-6s %d / %d = %4f" % ('train:', tp, total, accuracy))
            tp, total, accuracy = self.evaluate(devset, average=average)
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

    def update(self, batch):
        wiseq, tiseq = batch
        # 根据单词序列的正确词性更新权重
        for i, ti in enumerate(tiseq):
            # 根据现有权重向量预测词性
            pi = self.predict(wiseq, i)
            # 如果预测词性与正确词性不同，则更新权重
            if ti != pi:
                cfreqs = Counter(self.instantiate(wiseq, i, ti))
                efreqs = Counter(self.instantiate(wiseq, i, pi))
                fiseq, fcounts = map(list, zip(*[
                    (self.fdict[f], cfreqs[f] - efreqs[f])
                    for f in cfreqs | efreqs if f in self.fdict
                ]))

                # 累加权重加上步长乘以权重
                self.V[fiseq] += (self.k - self.R[fiseq]) * self.W[fiseq]
                # 更新权重
                self.W[fiseq] += fcounts
                # 更新时间戳记录
                self.R[fiseq] = self.k
                self.k += 1

    def predict(self, wiseq, index, average=False):
        fvs = [self.instantiate(wiseq, index, ti)
               for ti in range(self.nt)]
        scores = [self.score(fv, average=average) for fv in fvs]
        return np.argmax(scores)

    def score(self, fvector, average=False):
        # 计算特征对应累加权重的得分
        if average:
            scores = (self.V[self.fdict[f]]
                      for f in fvector if f in self.fdict)
        # 计算特征对应未累加权重的得分
        else:
            scores = (self.W[self.fdict[f]]
                      for f in fvector if f in self.fdict)
        return sum(scores)

    def instantiate(self, wiseq, index, ti):
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

    def evaluate(self, data, average=False):
        tp, total = 0, 0

        for wiseq, tiseq in data:
            total += len(wiseq)
            piseq = np.array([self.predict(wiseq, i, average)
                              for i in range(len(wiseq))])
            tp += np.sum(tiseq == piseq)
        accuracy = tp / total
        return tp, total, accuracy

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            lm = pickle.load(f)
        return lm
