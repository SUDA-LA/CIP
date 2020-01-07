# -*- coding: utf-8 -*-

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np

from bpnn import BPNN
from config import Config
from corpus import Corpus

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Back Propagation Neural Network(BPNN) '
                    'for POS Tagging.'
    )
    parser.add_argument('--adagrad', '-a',
                        action='store_true', default=False,
                        help='use adagrad')
    parser.add_argument('--threads', '-t',
                        action='store', default='8',
                        help='set the max num of threads')
    parser.add_argument('--seed', '-s',
                        action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--file', '-f',
                        action='store', default='bpnn.pkl',
                        help='set where to store the model')
    args = parser.parse_args()

    print("Set the max num of threads to %s\n"
          "Set the seed for generating random numbers to %d" %
          (args.threads, args.seed))
    os.environ['MKL_NUM_THREADS'] = args.threads
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 根据参数读取配置
    config = Config()

    print("Preprocess the data")
    # 以训练数据为基础建立语料
    corpus = Corpus(config.ftrain)
    # 用预训练词嵌入扩展语料并返回词嵌入矩阵
    embed = corpus.extend(config.embed)
    print(corpus)

    print("Load the dataset")
    trainset = corpus.load(config.ftrain, window=config.window)
    devset = corpus.load(config.fdev, window=config.window)
    testset = corpus.load(config.ftest, window=config.window)
    print("  size of trainset: %d\n"
          "  size of devset: %d" %
          (len(trainset), len(devset)))

    sizes = [config.embdim * config.window, *config.hidsizes, corpus.nt]

    start = datetime.now()

    print("Create Neural Network")
    print("  window: %d\n"
          "  vocdim: %d\n"
          "  sizes: " %
          (config.window, corpus.nw), sizes)
    bpnn = BPNN(window=config.window,
                vocdim=corpus.nw,
                sizes=sizes,
                embed=embed)

    print("Use SGD algorithm to trainset the network")
    print("  epochs: %d\n"
          "  batch_size: %d\n"
          "  interval: %d\n"
          "  eta: %f\n"
          "  lmbda: %f\n"
          "  adagrad: %r\n" %
          (config.epochs, config.batch_size, config.interval,
           config.eta, config.lmbda, args.adagrad))
    bpnn.SGD(trainset=trainset,
             devset=devset,
             file=args.file,
             epochs=config.epochs,
             batch_size=config.batch_size,
             interval=config.interval,
             eta=config.eta,
             lmbda=config.lmbda,
             adagrad=args.adagrad)

    bpnn = BPNN.load(args.file)
    loss = bpnn.loss(testset, config.lmbda)
    tp, total, accuracy = bpnn.evaluate(testset)
    print("%-6s Loss: %4f Accuracy: %d / %d = %4f" %
          ('test:', loss, tp, total, accuracy))
    print("%ss elapsed" % (datetime.now() - start))
