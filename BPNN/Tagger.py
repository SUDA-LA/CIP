from Reader.DataReader import DataReader
from Model.BiGruCrfModel import Model


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def train(self, dataset_path, config, test_path=None):
        trainer = Model.ModelTrainer(dataset_path, test_path=test_path, config=config)
        self.model = trainer.model
        trainer.train()

    def tag(self, sentence):
        return self.model.tag(sentence)

    def save(self, path):
        if self.model is not None:
            self.model.save(path)

    def load(self, path):
        self.model = Model.load(path)


if __name__ == '__main__':
    tagger = Tagger()
    tagger.train('./data/ctb5/train.conll',
                 test_path='./data/ctb5/test.conll',
                 config=Model.Config(hidden_dim=300, embedding_dim=100))
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


