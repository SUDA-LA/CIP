import abc
import pickle
from DataReader import DataReader


class TaggerBase(abc.ABC):
    def __init__(self, model_path=None):
        self.model_name = 'Linear Tagger'
        self.model = None
        if model_path:
            self.load_model(model_path)

    class Config:
        def __init__(self, stop_threshold=0,
                     max_iter=30,
                     check_point=None,
                     save_iter=5,
                     averaged_perceptron=False,
                     random_lr=False,
                     max_lr=1.2,
                     min_lr=0.8):
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.check_point = check_point
            self.save_iter = save_iter
            self.averaged_perceptron = averaged_perceptron
            self.random_lr = random_lr
            self.max_lr = max_lr
            self.min_lr = min_lr

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def evaluate(self, eval_path=None, eval_reader=None, averaged_perceptron=False):
        assert eval_path is not None or eval_reader is not None
        if eval_reader is None:
            eval_reader = DataReader(eval_path)
        s = eval_reader.get_seg_data()
        p = eval_reader.get_pos_data()
        right = 0
        word_count = 0
        for i, val in enumerate(s):
            tag = self.tag(val, averaged_perceptron=averaged_perceptron)
            right += len([index for index, v in enumerate(tag) if v == p[i][index]])
            word_count += len(tag)
        return right, word_count, right / word_count

    @abc.abstractmethod
    def train(self, data_path, test_path=None, dev_path=None, config=None):
        """
        :param data_path: the data path for train
        :param test_path: the data path for test (can be None)
        :param dev_path: the data path for dev (can be None)
        :param config:  the training config use Tagger.Config
        :return: Nothing
        """

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def tag(self, s, index=None, averaged_perceptron=False):
        assert self.model
        if index is None:
            return [self._tag(s, i, averaged_perceptron=averaged_perceptron) for i in range(len(s))]
        else:
            return self._tag(s, index, averaged_perceptron=averaged_perceptron)

    @abc.abstractmethod
    def _tag(self, s, index, averaged_perceptron=False):
        """
        :param s: input sentence
        :param index: the position that waiting to tag in sentence
        :param averaged_perceptron: if using averaged perceptron or not default is False
        :return: tag in index
        """

    @abc.abstractmethod
    def _dot(self, feature_vector, averaged_perceptron=False):
        """
        :param feature_vector: the feature vector if is n-D tensor the last dimension is feature
        :param averaged_perceptron: if using averaged perceptron or not default is False
        :return: (n-1)-D score for feature vector
        """

    @abc.abstractmethod
    def _get_feature_id(self, f, new_id=False):
        """
        :param f: feature
        :param new_id: add feature into or not when feature isn't in feature map
        :return: Nothing
        """

    @abc.abstractmethod
    def _extract_feature(self, *input):
        """
        :param input: depend on tagger type
        :return: feature that extracted
        """

    @abc.abstractmethod
    def _create_feature_space(self, segs, tags):
        """
        :param segs: sentence in train dataset
        :param tags: ground truth of tag in train dataset
        :return: Nothing
        """

