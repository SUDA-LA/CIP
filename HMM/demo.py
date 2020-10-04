from Tagger import Tagger
from config import Config
from DataReader import DataReader
import os
import argparse
import numpy as np

def main(args):
    # train
    tagger = Tagger()
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    model_path = os.path.join(args.path, 'model.pickle')
    train_path = os.path.join(args.data, 'train.conll')
    dev_path = os.path.join(args.data, 'dev.conll')
    if args.data == 'data':
        test_path = None
    elif args.data == 'bigdata':
        test_path = os.path.join(args.data, 'test.conll')
    else:
        raise ValueError()
    tagger.train(train_path,
                 dev_path=dev_path,
                 test_path=test_path,
                 config=args)  # small: 0.3 # big: 0.01
    tagger.save_model(model_path)
    tagger.load_model(model_path)

    # evaluate
    if args.data == 'data':
        dr = DataReader(dev_path)
    elif args.data == 'bigdata':
        dr = DataReader(test_path)
    else:
        raise ValueError()

    s = dr.get_seg_data()
    gt = dr.get_pos_data()
    acc = 0
    word_count = 0

    for i, val in enumerate(s):
        tag = tagger.tag(val)
        acc += len([index for index, v in enumerate(tag) if v == gt[i][index]])
        word_count += len(tag)

    print("Tagging Accuracy: %.5f" % (acc / word_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create HMM Tagger.')
    parser.add_argument('--data', default='data', choices=['data', 'bigdata'], help='path to train file')
    parser.add_argument('--path', '-p', default='model', help='path to model file')
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    main(args)

