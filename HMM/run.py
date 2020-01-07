# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np

from config import Config
from corpus import Corpus
from hmm import HMM

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Hidden Markov Model(HMM) for POS Tagging.'
    )
    parser.add_argument('--bigdata', '-b',
                        action='store_true', default=False,
                        help='use big data')
    parser.add_argument('--file', '-f',
                        action='store', default='hmm.pkl',
                        help='set where to store the model')
    args = parser.parse_args()

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

    print("Create HMM")
    hmm = HMM(corpus.nw, corpus.nt)

    print("Use %d sentences to train the HMM" % corpus.ns)
    hmm.train(trainset=trainset,
              alpha=config.alpha,
              file=args.file)

    print("Use Viterbi algorithm to tag the dataset")
    tp, total, accuracy = hmm.evaluate(devset)
    print("Accuracy of dev: %d / %d = %4f\n" % (tp, total, accuracy))

    if args.bigdata:
        hmm = HMM.load(args.file)
        tp, total, accuracy = hmm.evaluate(testset)
        print("Accuracy of test: %d / %d = %4f" % (tp, total, accuracy))

    print("%ss elapsed" % (datetime.now() - start))
