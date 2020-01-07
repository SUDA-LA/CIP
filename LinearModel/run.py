# -*- coding: utf-8 -*-

import argparse
import random
from datetime import datetime, timedelta

import numpy as np

from config import Config
from corpus import Corpus

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Linear Model(LM) for POS Tagging.'
    )
    parser.add_argument('--bigdata', '-b',
                        action='store_true', default=False,
                        help='use big data')
    parser.add_argument('--average', '-a',
                        action='store_true', default=False,
                        help='use average perceptron')
    parser.add_argument('--optimize', '-o',
                        action='store_true', default=False,
                        help='use feature extracion optimization')
    parser.add_argument('--seed', '-s',
                        action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--file', '-f',
                        action='store', default='lm.pkl',
                        help='set where to store the model')
    args = parser.parse_args()

    if args.optimize:
        from olm import LinearModel
    else:
        from lm import LinearModel

    print("Set the seed for generating random numbers to %d" % args.seed)
    random.seed(args.seed)

    # 根据参数读取配置
    config = Config(args.bigdata)

    print("Preprocess the data")
    corpus = Corpus(config.ftrain)
    print(corpus)

    print("Load the dataset")
    trainset = corpus.load(config.ftrain)
    devset = corpus.load(config.fdev)
    print("  size of trainset: %d\n"
          "  size of devset: %d" %
          (len(trainset), len(devset)))
    if args.bigdata:
        testset = corpus.load(config.ftest)
        print("  size of testset: %d" % len(testset))

    start = datetime.now()

    print("Create Linear Model")
    if args.optimize:
        print("  use feature extracion optimization")
    if args.average:
        print("  use average perceptron")
    lm = LinearModel(corpus.nt)

    print("Use %d sentences to create the feature space" % corpus.ns)
    lm.create_feature_space(trainset)
    print("The size of the feature space is %d" % lm.d)

    print("Use online-training algorithm to train the model")
    print("  epochs: %d\n"
          "  interval: %d\n" % (config.epochs, config.interval))
    lm.online(trainset=trainset,
              devset=devset,
              file=args.file,
              epochs=config.epochs,
              interval=config.interval,
              average=args.average)

    if args.bigdata:
        lm = LinearModel.load(args.file)
        print("Accuracy of test: %d / %d = %4f" %
              lm.evaluate(testset, average=args.average))

    print("%ss elapsed" % (datetime.now() - start))
