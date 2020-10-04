from config import Config
from Corpus import DataReader
import os
import argparse
import numpy as np
from Tagger import Tagger

def main(tagger, args):
    # train
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    model_path = os.path.join(args.path, 'model.pickle')
    train_path = os.path.join(args.data, 'train.conll')
    dev_path = os.path.join(args.data, 'dev.conll')
    test_path = os.path.join(args.data, 'test.conll')
    embedding_path = os.path.join(args.data, 'embed.txt')

    tagger.train(train_path,
                 dev_path=dev_path,
                 test_path=test_path,
                 embedding_path=embedding_path,
                 config=args)
    tagger.save_model(model_path)
    tagger.load_model(model_path)

    # evaluate
    tagger = Tagger(model_path)
    eval_dataset = tagger.data_reader.to_dataset(test_path)
    _, _, acc = tagger.evaluate(eval_dataset=eval_dataset)
    print("Tagging in epoch finish Accuracy: %.5f" % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create LinearModel Tagger.')
    parser.add_argument('--data', default='data', help='path to train file')
    parser.add_argument('--optimized', '-o', action='store_true',
                           help='whether to use optimized tagger')
    parser.add_argument('--path', '-p', default='model', help='path to model file')
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))

    main(Tagger(), args)
