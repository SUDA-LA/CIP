# -*- coding: utf-8 -*-


class Config(object):
    window = 5
    embdim = 100
    hidsizes = [300]
    epochs = 100
    batch_size = 50
    interval = 10
    eta = 0.5
    lmbda = 1
    ftrain = 'data/ctb5/train.conll'
    fdev = 'data/ctb5/dev.conll'
    ftest = 'data/ctb5/test.conll'
    embed = 'data/embed.txt'
