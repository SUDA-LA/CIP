import torch
from DataReader import DataReader
from Char2seqModel import Model


class Tagger:
    def __init__(self, config, model_path=None):
        self.config = config
        self.model = None

    def train(self, dataset_path):
        trainer = Model.ModelTrainer(dataset_path, self.config)
        self.model = trainer.model
        trainer.train()

    def tag(self, sentence):
        return self.model.tag(sentence)

    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == '__main__':
    tagger = Tagger(Model.Config(hidden_dim=20, embedding_dim=100))
    tagger.train('./data/ctb5/train.conll')
    tagger.save('./model/ctb5_model.model')
    dr = DataReader('./data/ctb5/test.conll')
    s = dr.get_seg_data()
    gt = dr.get_pos_data()

    acc = 0
    word_count = 0

    for i, val in enumerate(s):
        tag = tagger.tag(val)
        acc += len([index for index, v in enumerate(tag) if v == gt[i][index]])
        word_count += len(tag)

    print("Tagging Accuracy: %.5f" % acc / word_count)


