from config import Config
from DataReader import DataReader
import os
import argparse
import numpy as np

def main(tagger, args):
    # train
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

    tagger = Tagger(model_path)
    _, _, acc = tagger.evaluate(eval_reader=dr)
    print("Tagging in epoch finish Accuracy: %.5f" % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create LinearModel Tagger.')
    parser.add_argument('--data', default='data', choices=['data', 'bigdata'], help='path to train file')
    parser.add_argument('--optimized', '-o', action='store_true',
                           help='whether to use optimized tagger')
    parser.add_argument('--path', '-p', default='model', help='path to model file')
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    if args.optimized:
        from OptimizedTagger import Tagger
    else:
        from Tagger import Tagger
    main(Tagger(), args)
