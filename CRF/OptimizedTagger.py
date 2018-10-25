import pickle
import math
from DataReader import DataReader
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp


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
            self.tags = {}
            self.tags_backward = {}
            self.features = {}
            self.tag_size = 0
            self.feature_size = 0
            self.tag_weight = None

    class Config:
        def __init__(self, stop_threshold, max_iter, check_point=None, save_iter=5):
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.check_point = check_point
            self.save_iter = save_iter

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

        tag_g = np.zeros((self.model.tag_size, self.model.tag_size))
        g = np.zeros((self.model.tag_size, self.model.feature_size))
        batch_size = 50
        learning_rate = 20
        b = 0
        while not convergence:
            wrong = 0
            word_count = 0
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                features = [self._extract_feature(sentence, k) for k in range(s_len)] + [self._end_feature(sentence)]
                gt_pos = [int(t[p]) for p in pos[i]]
                scores, alpha, beta = self._get_scores(sentence)
                _, pred_pos_name = self._viterbi_decode(scores)
                z = np.average([alpha[s_len + 1][1], beta[0][0]])
                shape = beta.shape
                beta.shape = (shape[0], 1, shape[1])
                for k in range(s_len + 1):
                    gt_k = gt_pos[k] if k < s_len else 1
                    gt_k_1 = gt_pos[k - 1] if k > 0 else 0

                    tag_g[gt_k][gt_k_1] += 1
                    g[gt_k][features[k]] += 1

                    update = np.exp(alpha[k] + scores[k] + np.transpose(beta[k + 1]) - z)
                    # 0维向量无法转置
                    tag_g -= update
                    update_s = np.sum(update, axis=1)

                    for t_id in range(self.model.tag_size):
                        g[t_id][features[k]] -= update_s[t_id]

                    b += 1

                if b >= batch_size:
                    tag_weight += (learning_rate / batch_size) * tag_g
                    weight += (learning_rate / batch_size) * g

                    tag_g *= 0
                    g *= 0
                    b = 0

                word_count += s_len
                wrong += len([False for i, tag in enumerate(pos[i]) if tag != pred_pos_name[i]])

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

    def _viterbi_decode(self, scores):
        s_len = len(scores) - 1
        backwards = np.zeros((s_len + 1, self.model.tag_size))
        score = np.full((1, self.model.tag_size), -np.inf)
        score[0][0] = 0
        for i in range(s_len + 1):
            score_i = score + scores[i]
            backwards[i] = np.argmax(score_i, axis=1)
            score = np.max(score_i, axis=1)

        best_score, last_tag = score[1], backwards[-1][1]
        tag_point = int(last_tag)
        tags = [tag_point]
        for backward in reversed(backwards[:-1]):
            tag_point = int(backward[tag_point])
            tags.append(tag_point)
        start = tags.pop()
        assert start == 0
        return float(best_score), [self.model.tags_backward[t_id] for t_id in reversed(tags)]

    def _get_scores(self, s):
        assert self.model
        s_len = len(s)
        # 这里的alpha和beta准确地说是log(alpha)和log(beta)
        alpha = np.full((s_len + 2, self.model.tag_size), -np.inf)
        alpha[0][0] = 0
        beta = np.full((s_len + 2, self.model.tag_size), -np.inf)
        beta[s_len + 1][1] = 0
        t = self.model.tags
        tw = self.model.tag_weight
        t_b = self.model.tags_backward
        features = [self._extract_feature(s, i) for i in range(s_len)] + [self._end_feature(s)]
        # 开始的时候end_feature直接给的None (这个没什么毛病)
        scores = np.array([
            np.array([
                self._dot(now_tag, f) +
                np.array([
                    tw[now_tag][pre_tag]
                    for pre_tag in t.values()
                ])
                for now_tag in t.values()
            ])
            for i, f in enumerate(features)
        ])
        for k in range(s_len + 1):
            alpha[k + 1] = logsumexp(scores[k] + alpha[k], axis=1)
            beta[s_len - k] = logsumexp(np.transpose(scores[s_len - k]) + beta[s_len - k + 1], axis=1)
            # 开始的时候beta中的scores没有转置 (×)
        return scores, alpha, beta

    def tag(self, s):
        scores, _, _ = self._get_scores(s)
        _, tags = self._viterbi_decode(scores)
        return tags

    def _dot(self, now_tag, feature_vector):
        if feature_vector is None or len(feature_vector) == 0:
            return 0
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

    def _end_feature(self, s, new_id=False):
        wi = "$$"
        wim1 = s[-1]

        feature_vector = [self._get_feature_id((2, wi), new_id=new_id),
                          self._get_feature_id((3, wim1), new_id=new_id),
                          self._get_feature_id((5, wi, wim1[-1]), new_id=new_id),
                          self._get_feature_id((7, wi[0]), new_id=new_id),
                          self._get_feature_id((8, wi[-1]), new_id=new_id),
                          self._get_feature_id((13, wi[0]), new_id=new_id)]

        return np.array([feature for feature in feature_vector if feature is not None])

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
                feature_vector += [self._get_feature_id((13, wi[k]), new_id=new_id)]

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
        t["END"] = 1
        t_b[1] = "END"
        for i in range(data_len):
            sentence = segs[i]
            s_len = len(sentence)
            self._end_feature(sentence, new_id=True)
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
    # tagger.train('.\\bigdata\\train.conll', tagger.Config(0, 20, '.\\model\\', 1))
    tagger.train('.\\data\\train.conll', tagger.Config(0, 35, '.\\model\\', 5))
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
