# -*- coding: utf-8 -*-


class Corpus(object):
    UNK = '<UNK>'

    def __init__(self, fdata):
        # 获取数据的句子
        self.sentences = self.preprocess(fdata)
        # 获取数据的所有不同的词汇和词性
        self.words, self.tags = self.parse(self.sentences)
        # 增加未知词汇
        self.words += [self.UNK]

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        # 未知词汇索引
        self.ui = self.wdict[self.UNK]
        # 句子数量
        self.ns = len(self.sentences)
        # 词汇数量
        self.nw = len(self.words)
        # 词性数量
        self.nt = len(self.tags)

    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wiseq = [self.wdict.get(w, self.ui) for w in wordseq]
            tiseq = [self.tdict[t] for t in tagseq]
            data.append((wiseq, tiseq))
        return data

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
