import pickle
from .DataReader import DataReader
import numpy as np
from datetime import datetime


class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'HMM Tagger'
        self.model = None
        if model_path:
            self.load_model(model_path)

    class Model:
        def __init__(self):
            self.weight = None
            self.tags = {}
            self.tags_backward = {}
            self.features = {0: 0}
            self.tag_size = 0
            self.feature_size = 0
            self.tag_weight = None
            self.smooth = 0
            self.START_TAG = None
            self.STOP_TAG = None

    class Config:
        def __init__(self, smooth=1., check_point=None, evaluate_mode=False):
            self.smooth = smooth
            self.check_point = check_point
            self.evaluate_mode = evaluate_mode

    def load_model(self, model_path):
        self.model = None
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def evaluate(self, eval_path=None, eval_reader=None):
        assert eval_path is not None or eval_reader is not None
        if eval_reader is None:
            eval_reader = DataReader(eval_path)
        s = eval_reader.get_seg_data()
        p = eval_reader.get_pos_data()
        right = 0
        word_count = 0
        for i, val in enumerate(s):
            tag = self.tag(val)
            right += len([index for index, v in enumerate(tag) if v == p[i][index]])
            word_count += len(tag)

        return right, word_count, right / word_count

    def train(self, data_path, test_path=None, dev_path=None, config=None):
        if config is None:
            config = self.Config()

        smooth = config.smooth
        check_point = config.check_point
        evaluate_mode = config.evaluate_mode

        if evaluate_mode:
            dr = DataReader(data_path, random_seed=1)
            print(f"Set the seed for built-in generating random numbers to 1")
            np.random.seed(1)
            print(f"Set the seed for numpy generating random numbers to 1")
        else:
            dr = DataReader(data_path)
        if test_path is None:
            test_reader = None
        else:
            test_reader = DataReader(test_path)
        if dev_path is None:
            dev_reader = None
        else:
            dev_reader = DataReader(dev_path)

        self.model = self.Model()
        self.model.smooth = smooth
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_count = len(data)
        self._create_feature_space(data, pos)
        weight = self.model.weight
        tag_weight = self.model.tag_weight
        t = self.model.tags
        START_TAG = self.model.START_TAG
        extract_feature = self._extract_feature
        start = datetime.now()

        for i in range(data_count):
            sentence = data[i]
            gt_pos = np.array([int(t[p]) for p in pos[i]])
            s_len = len(data[i])
            features = np.array([
                extract_feature(sentence, k)
                for k in range(s_len)
            ])
            tag_feature = [
                (gt_pos[k - 1] if k > 0 else START_TAG, gt_pos[k]) for k in range(s_len)
            ]
            for k in range(s_len):
                weight[gt_pos[k]][features[k]] += 1
                tag_weight[tag_feature[k]] += 1

        tag_count = np.sum(tag_weight, axis=1)  # tag_w: (now, next) -> tag_c : (now)
        features_count = np.sum(weight, axis=0)  # tag_w: (tag, feature) -> tag_c : (feature)
        # tag_count[START_TAG] = data_count
        self.model.weight = (weight + smooth) / (features_count.reshape(1, -1) + smooth * self.model.tag_size)
        # (tag, feature) / (1, feature)
        self.model.tag_weight = (tag_weight + smooth) / (tag_count.reshape(-1, 1) + smooth * self.model.tag_size)
        # (now, next) / (now, 1)

        _, _, train_acc = self.evaluate(eval_reader=dr)
        print(f"train accuracy: {train_acc :.5%}")
        if dev_reader is not None:
            _, _, dev_acc = self.evaluate(eval_reader=dev_reader)
            print(f"dev   accuracy: {dev_acc :.5%}")
        if test_reader is not None:
            _, _, test_acc = self.evaluate(eval_reader=test_reader)
            print(f"test  accuracy: {test_acc :.5%}")

        spend = datetime.now() - start
        if check_point:
            self.save_model(check_point + 'check_point_finish.pickle')
        print(f"spend time: {spend}s\n")

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def tag(self, s):
        assert self.model
        s_len = len(s)
        pi = np.zeros((s_len + 1, self.model.tag_size))
        bt = np.zeros((s_len, self.model.tag_size))
        pi[0][0] = 1
        tw = self.model.tag_weight.transpose()
        for i in range(s_len):
            f = self._extract_feature(s, i)
            scores = self._dot(f).reshape((-1, 1)) * tw * pi[i]
            # (now_tag, 1) + (now_tag, pre_tag) + (1, pre_tag)
            bt[i] = np.argmax(scores, axis=1)
            pi[i + 1] = np.max(scores, axis=1)
        tags = [''] * s_len
        pre_tag = np.argmax(pi[-1])
        for i in range(s_len):
            index = s_len - i - 1
            now_tag = int(pre_tag)
            tags[index] = self.model.tags_backward[now_tag]
            pre_tag = bt[index][now_tag]
        return tags

    def _dot(self, feature_vector):
        return self.model.weight[:, feature_vector]

    def _get_feature_id(self, f, new_id=False):
        feature_map = self.model.features
        if f in feature_map:
            return feature_map[f]
        else:
            if new_id:
                feature_id = len(feature_map)
                feature_map[f] = feature_id
                return feature_id
            else:
                return 0

    def _extract_feature(self, s, index, new_id=False):
        return self._get_feature_id(s[index], new_id=new_id)

    def _create_feature_space(self, segs, tags):
        data_len = len(segs)
        t = self.model.tags
        t_b = self.model.tags_backward
        t["START"] = 0
        t_b[0] = "START"
        self.model.START_TAG = 0
        for i in range(data_len):
            sentence = segs[i]
            s_len = len(sentence)
            for k in range(s_len):
                tag = tags[i][k]
                self._extract_feature(sentence, k, new_id=True)
                if tag not in t:
                    t_id = len(t)
                    t[tag] = t_id
                    t_b[t_id] = tag
        self.model.tag_size = len(t)
        self.model.feature_size = len(self.model.features)
        self.model.tag_weight = np.zeros((self.model.tag_size, self.model.tag_size))
        self.model.weight = np.zeros((self.model.tag_size, self.model.feature_size))


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll',
                 dev_path='.\\data\\dev.conll',
                 #test_path='.\\bigdata\\test.conll',
                 config=tagger.Config(0.3, '.\\model\\'))  # small: 0.3 # big: 0.01
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
