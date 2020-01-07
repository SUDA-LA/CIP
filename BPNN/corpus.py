# -*- coding: utf-8 -*-

import numpy as np


class Corpus(object):
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

    def __init__(self, fdata):
        # 获取数据的句子
        self.sentences = self.preprocess(fdata)
        # 获取数据的所有不同的词汇和词性
        self.words, self.tags = self.parse(self.sentences)
        # 增加句首词汇、句尾词汇和未知词汇
        self.words += [self.SOS, self.EOS, self.UNK]

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}

        # 句首词汇索引
        self.si = self.wdict[self.SOS]
        # 句尾词汇索引
        self.ei = self.wdict[self.EOS]
        # 未知词汇索引
        self.ui = self.wdict[self.UNK]

        # 句子数量
        self.ns = len(self.sentences)
        # 词汇数量
        self.nw = len(self.words)
        # 词性数量
        self.nt = len(self.tags)

    def extend(self, fembed):
        with open(fembed, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        # 获取预训练数据中的词汇和嵌入矩阵
        words, embed = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        unk_words = [w for w in words if w not in self.wdict]
        # 扩展词汇
        self.words = sorted(set(self.words + unk_words))
        # 更新字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 更新索引
        self.si = self.wdict[self.SOS]
        self.ei = self.wdict[self.EOS]
        self.ui = self.wdict[self.UNK]
        # 更新词汇数
        self.nw = len(self.words)
        # 初始化词嵌入矩阵
        embed = np.array(embed)
        vocdim, embdim = self.nw, embed.shape[1]
        indices = [self.wdict[w] for w in words]
        # 在预训练矩阵中的词采用预训练的词向量，否则随机初始化
        extended_embed = np.random.randn(vocdim, embdim) / np.sqrt(embdim)
        extended_embed[indices] = embed
        return extended_embed

    def load(self, fdata, window=5):
        data = []
        half = window // 2
        sentences = self.preprocess(fdata)
        for wordseq, tagseq in sentences:
            wiseq = [self.wdict.get(w, self.ui) for w in wordseq]
            wiseq = [self.si] * half + wiseq + [self.ei] * half
            tiseq = [self.tdict[t] for t in tagseq]
            for i, ti in enumerate(tiseq):
                x = wiseq[i:i + window]
                y = self.vectorize(ti)
                data.append((x, y))
        return data

    def vectorize(self, i):
        e = np.zeros((self.nt, 1))
        e[i] = 1.0
        return e

    def __repr__(self):
        info = "%s(\n" % self.__class__.__name__
        info += "  num of sentences: %d\n" % self.ns
        info += "  num of words: %d\n" % self.nw
        info += "  num of tags: %d\n" % self.nt
        info += ")"
        return info

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []
        with open(fdata, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))
        return sentences

    @staticmethod
    def parse(sentences):
        wordseqs, tagseqs = zip(*sentences)
        words = sorted(set(w for wordseq in wordseqs for w in wordseq))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        return words, tags
