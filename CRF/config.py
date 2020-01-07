# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, bigdata=False):
        self.epochs = 100
        self.lmbda = 0.01
        self.decay = 0.96
        if bigdata:
            self.batch_size = 50
            self.eta = 0.2
            self.interval = 10
            self.ftrain = 'bigdata/train.conll'
            self.fdev = 'bigdata/dev.conll'
            self.ftest = 'bigdata/test.conll'
        else:
            self.batch_size = 1
            self.eta = 0.5
            self.interval = 5
            self.ftrain = 'data/train.conll'
            self.fdev = 'data/dev.conll'
