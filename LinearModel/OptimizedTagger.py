import pickle
from DataReader import DataReader


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def train(self, data_path):
        dr = DataReader(data_path)
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self.model = {'weight': {}, 'tags': {}, 'v': {}, 'feature_map': {}, 'g': 0}
        self._create_feature_space(data, pos)
        convergence = False
        stop_threshold = 0.0001
        max_iter = 20
        iter_count = 0
        learning_rate = 1
        rho = 0.9
        update_stamp = {}
        step = 0
        t = self.model['tags']
        g = self.model['g']
        while not convergence:
            wrong_count = 0
            word_count = 0
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                word_count += s_len
                for k in range(s_len):
                    predict_tag = self.tag(sentence, k)
                    tag = pos[i][k]
                    f_raw = self._extract_feature(sentence, k)
                    if predict_tag != tag:
                        fp = [t[predict_tag] * g + fi for fi in f_raw]
                        fgt = [t[tag] * g + fi for fi in f_raw]
                        for f in fp:
                            if f in update_stamp:
                                self.model['v'][f] += (step - update_stamp[f] - 1) * self.model['weight'][f]
                            else:
                                self.model['v'][f] = 0
                            update_stamp[f] = step
                            self.model['weight'][f] = self.model['weight'].setdefault(f, 0) - learning_rate
                            self.model['v'][f] += self.model['weight'][f]
                        for f in fgt:
                            if f in update_stamp:
                                self.model['v'][f] += (step - update_stamp[f] - 1) * self.model['weight'][f]
                            else:
                                self.model['v'][f] = 0
                            update_stamp[f] = step
                            self.model['weight'][f] = self.model['weight'].setdefault(f, 0) + learning_rate
                            self.model['v'][f] += self.model['weight'][f]
                        wrong_count += 1

            loss = wrong_count / word_count
            if loss < stop_threshold or iter_count > max_iter:
                convergence = True
                step += 1
                for f, stamp in update_stamp.items():
                    self.model['v'][f] += (step - update_stamp[f] - 1) * self.model['weight'][f]
                print("train finish loss: %.6f" % loss)
            else:
                step += 1
                iter_count += 1
                learning_rate *= rho
                print("iter: %d loss: %.6f" % (iter_count, loss))

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def tag(self, s, index=None, averaged_perceptron=False):
        assert self.model
        if index is None:
            s_len = len(s)
            tags = []
            for i in range(s_len):
                tags.append(self._tag(s, i, averaged_perceptron))
            return tags
        else:
            return self._tag(s, index, averaged_perceptron)

    def _tag(self, s, index, averaged_perceptron=False):
        max_tag = ''
        max_score = float('-Inf')
        f = self._extract_feature(s, index)
        t = self.model['tags']
        g = self.model['g']
        for tag, tag_id in t.items():
            fv = [tag_id * g + fi for fi in f]
            score = self._dot(fv, averaged_perceptron)
            if score > max_score:
                max_score = score
                max_tag = tag
        return max_tag

    def _dot(self, feature_vector, averaged_perceptron=False):
        score = 0
        if averaged_perceptron:
            weight = "v"
        else:
            weight = "weight"
        for f in feature_vector:
            score += self.model[weight].get(f, 0)
        return score

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

    def _extract_feature(self, s, index):
        feature_vector = [self._get_feature_id((2, s[index]), new_id=True)]

        wi = s[index]

        if index > 0:
            wim1 = s[index - 1]
        else:
            wim1 = "^^"

        if index < len(s) - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"

        feature_vector.append(self._get_feature_id((3, wim1), new_id=True))
        feature_vector.append(self._get_feature_id((4, wip1), new_id=True))
        feature_vector.append(self._get_feature_id((5, wi, wim1[-1]), new_id=True))
        feature_vector.append(self._get_feature_id((6, wi, wip1[0]), new_id=True))

        feature_vector.append(self._get_feature_id((7, wi[0]), new_id=True))
        feature_vector.append(self._get_feature_id((8, wi[-1]), new_id=True))

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector.append(self._get_feature_id((9, wi[k]), new_id=True))
            feature_vector.append(self._get_feature_id((10, wi[0], wi[k]), new_id=True))
            feature_vector.append(self._get_feature_id((11, wi[-1], wi[k]), new_id=True))

        if w_len == 1:
            feature_vector.append(self._get_feature_id((12, wi, wim1[-1], wip1[0]), new_id=True))

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector.append(self._get_feature_id((13, wi[k], "__C0nsecut1ve?__"), new_id=True))

        for k in range(1, min(5, w_len + 1)):
            feature_vector.append(self._get_feature_id((14, wi[:k]), new_id=True))
            feature_vector.append(self._get_feature_id((15, wi[-k:]), new_id=True))

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
                fv = self._extract_feature(sentence, k)
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
    tagger.train('.\\data\\train.conll')
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
