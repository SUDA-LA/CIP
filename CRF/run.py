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
        description='Create Conditional Random Field(CRF) for POS Tagging.'
    )
    parser.add_argument('--bigdata', '-b',
                        action='store_true', default=False,
                        help='use big data')
    parser.add_argument('--anneal', '-a',
                        action='store_true', default=False,
                        help='use simulated annealing')
    parser.add_argument('--optimize', '-o',
                        action='store_true', default=False,
                        help='use feature extracion optimization')
    parser.add_argument('--regularize', '-r',
                        action='store_true', default=False,
                        help='use L2 regularization')
    parser.add_argument('--seed', '-s',
                        action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--file', '-f',
                        action='store', default='crf.pkl',
                        help='set where to store the model')
    args = parser.parse_args()

    if args.optimize:
        from ocrf import CRF
    else:
        from crf import CRF

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

    print("Create Conditional Random Field")
    if args.optimize:
        print("  use feature extracion optimization")
    if args.anneal:
        print("  use simulated annealing")
    if args.regularize:
        print("  use L2 regularization")
    crf = CRF(corpus.nt)

    print("Use %d sentences to create the feature space" % corpus.ns)
    crf.create_feature_space(trainset)
    print("The size of the feature space is %d" % crf.d)

    print("Use SGD algorithm to train the model")
    print("  epochs: %d\n"
          "  batch_size: %d\n"
          "  interval: %d\n"
          "  eta: %f" %
          (config.epochs, config.batch_size,  config.interval, config.eta))
    if args.anneal:
        print("  dacay: %f" % config.decay)
    if args.regularize:
        print("  lmbda: %f" % config.lmbda)
    print()
    crf.SGD(trainset=trainset,
            devset=devset,
            file=args.file,
            epochs=config.epochs,
            batch_size=config.batch_size,
            interval=config.interval,
            eta=config.eta,
            decay=config.decay,
            lmbda=config.lmbda,
            anneal=args.anneal,
            regularize=args.regularize)

    if args.bigdata:
        crf = CRF.load(args.file)
        print("Accuracy of test: %d / %d = %4f" % crf.evaluate(testset))

    print("%ss elapsed" % (datetime.now() - start))
