from Reader.DataReader import DataReader
from Model.BiGruCrfModel import Model


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def train(self, data_reader: DataReader, config):
        trainer = Model.ModelTrainer(data_reader=data_reader, config=config)
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
    device = "cpu"
    data_reader = DataReader(
        train_path='./data/ctb5/train.conll',
        test_path='./data/ctb5/test.conll',
        dev_path='./data/ctb5/dev.conll',
        embed_path='./data/embed.txt',
        char_level=False,
        device=device
    )
    tagger.train(data_reader=data_reader, config=Model.Config(300, learn_rate=0.001, device=device))
    tagger.save('./model/ctb5_model.model')


