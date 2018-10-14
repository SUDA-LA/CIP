import pickle
import math
from DataReader import DataReader
from collections import defaultdict
import numpy as np
import scipy.misc


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        self.test_reader = None
        self.test_path = None
        if model_path:
            self.load_model(model_path)

    class Model:
        def __init__(self):
            self.weight = None
            self.v = None
            self.tags = {}
            self.tags_backward = {}
            self.features = {}
            self.tag_size = 0
            self.feature_size = 0
            self.tag_weight = None
            self.tag_v = None

    class Config:
        def __init__(self, stop_threshold, max_iter, check_point=None, save_iter=5):
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.check_point = check_point
            self.save_iter=save_iter

    def load_model(self, model_path):
        self.model = None
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def test(self, test_path=None):
        if not self.test_reader or test_path != self.test_path:
            self.test_reader = DataReader(test_path)
            self.test_path = test_path
        assert self.test_reader is not None or self.test_path is not None
        s = self.test_reader.get_seg_data()
        p = self.test_reader.get_pos_data()
        wrong = 0
        word_count = 0
        for i, val in enumerate(s):
            tag = self.tag(val)
            wrong += len([index for index, v in enumerate(tag) if v != p[i][index]])
            word_count += len(tag)

        return wrong / word_count

    def train(self, data_path, config=None):
        dr = DataReader(data_path)
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self.model = self.Model()
        self._create_feature_space(data, pos)
        weight = self.model.weight
        tag_v = self.model.tag_v
        v = self.model.v
        t = self.model.tags

        convergence = False
        if config is None:
            stop_threshold = 0
            max_iter = 35
            check_point = None
            save_iter = 5
        else:
            stop_threshold = config.stop_threshold
            max_iter = config.max_iter
            check_point = config.check_point
            save_iter = config.save_iter

        iter_count = 0
        tag_weight = self.model.tag_weight
        while not convergence:
            wrong = 0
            word_count = 0
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                features = [self._extract_feature(sentence, k) for k in range(s_len)]
                gt_pos = [int(t[p]) for p in pos[i]]
                pred_pos = [int(t[p]) for p in self.tag(sentence)]
                if gt_pos != pred_pos:
                    p_rate = 0.8 + 0.4 * np.random.random()
                    n_rate = 0.8 + 0.4 * np.random.random()
                    # p_rate = 1
                    # n_rate = 1
                    for k in range(s_len):
                        gt_k = gt_pos[k]
                        gt_k_1 = gt_pos[k - 1] if k > 0 else 0
                        pred_k = pred_pos[k]
                        pred_k_1 = pred_pos[k - 1] if k > 0 else 0

                        tag_weight[gt_k][gt_k_1] += p_rate
                        weight[gt_k][features[k]] += p_rate
                        tag_weight[pred_k][pred_k_1] -= n_rate
                        weight[pred_k][features[k]] -= n_rate

                    tag_v += tag_weight
                    v += weight

                word_count += s_len
                wrong += len([False for i, tag in enumerate(gt_pos) if tag != pred_pos[i]])

            loss = wrong / word_count
            # loss = self.test('.\\data\\dev.conll')
            if loss < stop_threshold or iter_count >= max_iter:
                convergence = True
                if check_point:
                    self.save_model(check_point + 'check_point_finish.pickle')
                print("train finish loss: %.6f" % loss)
            else:
                iter_count += 1
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(check_point + 'check_point_' + str(iter_count) + '.pickle')
                print("iter: %d loss: %.6f" % (iter_count, loss))

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def tag(self, s, averaged_perceptron=False):
        assert self.model
        s_len = len(s)
        pi = np.full((s_len + 1, self.model.tag_size), -np.inf)
        bt = np.zeros((s_len, self.model.tag_size))
        pi[0][0] = 0
        t = self.model.tags
        if averaged_perceptron:
            tw = self.model.tag_v
        else:
            tw = self.model.tag_weight
        for i in range(s_len):
            f = self._extract_feature(s, i)
            scores = np.array([
                self._dot(now_tag, f, averaged_perceptron=averaged_perceptron) +
                np.array([
                    tw[now_tag][pre_tag] + pi[i][pre_tag]
                    for pre_tag in t.values()
                ])
                for now_tag in t.values()
            ])
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

    def _dot(self, now_tag, feature_vector, averaged_perceptron=False):
        if len(feature_vector) == 0:
            return 0
        else:
            if averaged_perceptron:
                return np.sum(self.model.v[now_tag][feature_vector])
            else:
                return np.sum(self.model.weight[now_tag][feature_vector])

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
                return None

    def _extract_feature(self, s, index, new_id=False):
        wi = s[index]

        if index > 0:
            wim1 = s[index - 1]
        else:
            wim1 = "^^"

        if index < len(s) - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"
        feature_vector = [self._get_feature_id((2, s[index]), new_id=new_id),
                          self._get_feature_id((3, wim1), new_id=new_id),
                          self._get_feature_id((4, wip1), new_id=new_id),
                          self._get_feature_id((5, wi, wim1[-1]), new_id=new_id),
                          self._get_feature_id((6, wi, wip1[0]), new_id=new_id),
                          self._get_feature_id((7, wi[0]), new_id=new_id),
                          self._get_feature_id((8, wi[-1]), new_id=new_id)]

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector += [self._get_feature_id((9, wi[k]), new_id=new_id),
                                   self._get_feature_id((10, wi[0], wi[k]), new_id=new_id),
                                   self._get_feature_id((11, wi[-1], wi[k]), new_id=new_id)]

        if w_len == 1:
            feature_vector += [self._get_feature_id((12, wi, wim1[-1], wip1[0]), new_id=new_id)]

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector += [self._get_feature_id((13, wi[k], "__C0nsecut1ve?__"), new_id=new_id)]

        for k in range(1, min(5, w_len + 1)):
            feature_vector += [self._get_feature_id((14, wi[:k]), new_id=new_id),
                                   self._get_feature_id((15, wi[-k:]), new_id=new_id)]

        return np.array([feature for feature in feature_vector if feature is not None])

    def _create_feature_space(self, segs, tags):
        data_len = len(segs)
        t = self.model.tags
        t_b = self.model.tags_backward
        t["START"] = 0
        t_b[0] = "START"
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
        self.model.v = np.zeros((self.model.tag_size, self.model.feature_size))
        self.model.tag_v = np.zeros((self.model.tag_size, self.model.tag_size))


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    # tagger.train('.\\bigdata\\train.conll', tagger.Config(0, 20, '.\\model\\', 1))
    tagger.train('.\\data\\train.conll', tagger.Config(0, 35, '.\\model\\', 5))
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
