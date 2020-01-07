# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self, bigdata=False):
        if bigdata:
            self.alpha = 0.01  # 平滑参数
            self.ftrain = 'bigdata/train.conll'
            self.fdev = 'bigdata/dev.conll'
            self.ftest = 'bigdata/test.conll'
        else:
            self.alpha = 0.3  # 平滑参数
            self.ftrain = 'data/train.conll'
            self.fdev = 'data/dev.conll'
