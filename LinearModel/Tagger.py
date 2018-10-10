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
        self.model = {'weight': {}, 'tags': set()}
        self._create_feature_space(data, pos)
        convergence = False
        stop_threshold = 0.0001
        max_iter = 20
        iter_count = 0
        learning_rate = 1
        rho = 0.9
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
                    if predict_tag != tag:
                        fp = self._extract_feature(sentence, k, predict_tag)
                        fgt = self._extract_feature(sentence, k, tag)
                        for f in fp:
                            self.model['weight'][f] = self.model['weight'].setdefault(f, 0) - learning_rate
                        for f in fgt:
                            self.model['weight'][f] += learning_rate
                        wrong_count += 1

            loss = wrong_count / word_count
            if loss < stop_threshold or iter_count > max_iter:
                convergence = True
                print("train finish loss: %.6f" % loss)
            else:
                iter_count += 1
                learning_rate *= rho
                print("iter: %d loss: %.6f" % (iter_count, loss))

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

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
        max_tag = ''
        max_score = 0
        for tag in self.model["tags"]:
            fv = self._extract_feature(s, index, tag)
            score = self._dot(fv)
            if score > max_score:
                max_score = score
                max_tag = tag
        return max_tag

    def _dot(self, feature_vector):
        score = 0
        for f in feature_vector:
            score += self.model["weight"].get(f, 0)
        return score

    @staticmethod
    def _extract_feature(s, index, tag):
        feature_vector = ["__@?__".join(["02:" + tag, s[index]])]
        wi = s[index]

        if index > 0:
            wim1 = s[index - 1]
        else:
            wim1 = "^^"

        if index < len(s) - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"

        feature_vector.append("__@?__".join(["03:" + tag, wim1]))
        feature_vector.append("__@?__".join(["04:" + tag, wip1]))
        feature_vector.append("__@?__".join(["05:" + tag, wi, wim1[-1]]))
        feature_vector.append("__@?__".join(["06:" + tag, wi, wip1[0]]))

        feature_vector.append("__@?__".join(["07:" + tag, wi[0]]))
        feature_vector.append("__@?__".join(["08:" + tag, wi[-1]]))

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector.append("__@?__".join(["09:" + tag, wi[k]]))
            feature_vector.append("__@?__".join(["10:" + tag, wi[0], wi[k]]))
            feature_vector.append("__@?__".join(["11:" + tag, wi[-1], wi[k]]))

        if w_len == 1:
            feature_vector.append("__@?__".join(["12:" + tag, wi, wim1[-1], wip1[0]]))

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector.append("__@?__".join(["13:" + tag, wi[k], "__C0nsecut1ve?__"]))

        for k in range(min(5, w_len + 1)):
            feature_vector.append("__@?__".join(["14:" + tag, wi[:k]]))
            feature_vector.append("__@?__".join(["15:" + tag, wi[-k:]]))

        return feature_vector

    def _create_feature_space(self, segs, tags):
        data_len = len(segs)
        for i in range(data_len):
            sentence = segs[i]
            s_len = len(sentence)
            for k in range(s_len):
                tag = tags[i][k]
                fv = self._extract_feature(sentence, k, tag)
                for f in fv:
                    if f not in self.model['weight']:
                        self.model['weight'][f] = 0
                if tag not in self.model['tags']:
                    self.model['tags'].add(tag)


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll')
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
