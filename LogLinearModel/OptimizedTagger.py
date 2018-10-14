import pickle
import math
from DataReader import DataReader
import scipy.misc


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        self.test_reader = None
        self.test_path = None
        if model_path:
            self.load_model(model_path)

    class Config:
        def __init__(self, learning_rate, c, rho, stop_threshold, max_iter, batch_size):
            self.learning_rate = learning_rate
            self.c = c
            self.rho = rho
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.batch_size = batch_size

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
        acc = 0
        word_count = 0
        for i, val in enumerate(s):
            tag = self.tag(val)
            acc += len([index for index, v in enumerate(tag) if v == p[i][index]])
            word_count += len(tag)

        return 1 - (acc / word_count)

    def train(self, data_path, test_path=None, check_point=None, config=None):
        dr = DataReader(data_path)
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self.model = {'weight': {}, 'tags': {}, 'feature_map': {}, 'g': 0}
        self._create_feature_space(data, pos)
        convergence = False
        if config is None:
            batch_size = 50
            stop_threshold = 0.0001
            max_iter = 20
            learning_rate = 1
            c = 0.001
            rho = 0.9
        else:
            learning_rate = config.learning_rate
            c = config.c
            rho = config.rho
            stop_threshold = config.stop_threshold
            max_iter = config.max_iter
            batch_size = config.batch_size

        iter_count = 0
        t = self.model['tags']
        tags = [tag for tag in t]
        weight = self.model['weight']
        g = self.model['g']
        dw = {}
        b = 0
        while not convergence:
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                for k in range(s_len):
                    f_raw = self._extract_feature(sentence, k)
                    features = [[f + g * t[tag] for f in f_raw] for tag in tags]
                    ewf = [math.pow(math.e, min(self._dot(feature), 500)) for feature in features]
                    ewf_sum = sum(ewf)
                    for f_i, feature in enumerate(features):
                        for f in feature:
                            if f in dw:
                                dw[f] -= ewf[f_i] / ewf_sum
                            else:
                                dw[f] = -ewf[f_i] / ewf_sum
                        if tags[f_i] == pos[i][k]:
                            for f in feature:
                                if f in dw:
                                    dw[f] += 1
                                else:
                                    dw[f] = 1
                    b += 1
                    if b == batch_size:
                        for f, val in dw.items():
                            if f in weight:
                                weight[f] = (1 - c) * weight[f] + learning_rate * val / batch_size
                            else:
                                weight[f] = learning_rate * val / batch_size
                        b = 0
                        dw = {}
                    pass
            if test_path is None:
                self.test_reader = dr
                loss = self.test()
            else:
                loss = self.test(test_path)
            if loss < stop_threshold or iter_count >= max_iter:
                convergence = True
                print("train finish loss: %.6f" % loss)
            else:
                iter_count += 1
                learning_rate *= rho
                print("iter: %d loss: %.6f" % (iter_count, loss))

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def tag(self, s, index=None):
        assert self.model
        if index is None:
            s_len = len(s)
            tags = []
            for i in range(s_len):
                tags.append(self._tag(s, i))
            return tags
        else:
            return self._tag(s, index)

    def _tag(self, s, index):
        f = self._extract_feature(s, index)
        t = self.model['tags']
        g = self.model['g']
        tags = [tag for tag in t]
        score = [self._dot([tag_id * g + fi for fi in f]) for tag_id in t.values()]
        return tags[score.index(max(score))]

    def _dot(self, feature_vector):
        weight = "weight"
        scores = [self.model[weight].get(f, 0) for f in feature_vector]
        return sum(scores)

    def _get_feature_id(self, f, new_id=False):
        feature_map = self.model['feature_map']
        if f in feature_map:
            return feature_map[f]
        else:
            if new_id:
                feature_id = len(feature_map)
                feature_map[f] = feature_id
                return feature_id
            else:
                return -1

    def _extract_feature(self, s, index, new_id=False):
        feature_vector = [self._get_feature_id((2, s[index]), new_id=new_id)]

        wi = s[index]

        if index > 0:
            wim1 = s[index - 1]
        else:
            wim1 = "^^"

        if index < len(s) - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"

        feature_vector.append(self._get_feature_id((3, wim1), new_id=new_id))
        feature_vector.append(self._get_feature_id((4, wip1), new_id=new_id))
        feature_vector.append(self._get_feature_id((5, wi, wim1[-1]), new_id=new_id))
        feature_vector.append(self._get_feature_id((6, wi, wip1[0]), new_id=new_id))

        feature_vector.append(self._get_feature_id((7, wi[0]), new_id=new_id))
        feature_vector.append(self._get_feature_id((8, wi[-1]), new_id=new_id))

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector.append(self._get_feature_id((9, wi[k]), new_id=new_id))
            feature_vector.append(self._get_feature_id((10, wi[0], wi[k]), new_id=new_id))
            feature_vector.append(self._get_feature_id((11, wi[-1], wi[k]), new_id=new_id))

        if w_len == 1:
            feature_vector.append(self._get_feature_id((12, wi, wim1[-1], wip1[0]), new_id=new_id))

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector.append(self._get_feature_id((13, wi[k], "__C0nsecut1ve?__"), new_id=new_id))

        for k in range(1, min(5, w_len + 1)):
            feature_vector.append(self._get_feature_id((14, wi[:k]), new_id=new_id))
            feature_vector.append(self._get_feature_id((15, wi[-k:]), new_id=new_id))

        return feature_vector

    def _create_feature_space(self, segs, tags):
        data_len = len(segs)
        w = self.model['weight']
        t = self.model['tags']
        for i in range(data_len):
            sentence = segs[i]
            s_len = len(sentence)
            for k in range(s_len):
                tag = tags[i][k]
                fv = self._extract_feature(sentence, k, new_id=True)
                for f in fv:
                    if f not in w:
                        w[f] = 0
                if tag not in t:
                    t[tag] = len(t)
        self.model['g'] = len(w)


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll', '.\\data\\dev.conll', config=Tagger.Config(learning_rate=25,
                                                                                    c=0.001,
                                                                                    rho=0.95,
                                                                                    stop_threshold=0.001,
                                                                                    max_iter=50,
                                                                                    batch_size=50))
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
