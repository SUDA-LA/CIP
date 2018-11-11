import pickle
from datetime import datetime
from datetime import timedelta
from .DataReader import DataReader
import numpy as np
from scipy.special import logsumexp


class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'Log Linear Tagger'
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

    class Config:
        def __init__(self,
                     learning_rate=0.5,
                     c=0.,
                     rho=1.,
                     delay_step=100000,
                     max_iter=30,
                     batch_size=50,
                     check_point=None,
                     save_iter=5,
                     evaluate_mode=False):
            self.learning_rate = learning_rate
            self.c = c
            self.rho = rho
            self.max_iter = max_iter
            self.batch_size = batch_size
            self.check_point = check_point
            self.save_iter = save_iter
            self.delay_step = delay_step
            self.evaluate_mode = evaluate_mode

    def load_model(self, model_path):
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
            gt = p[i]
            right += len([index for index, v in enumerate(gt) if self._tag(val, index) == gt[index]])
            word_count += len(gt)

        return right, word_count, right / word_count

    def train(self, data_path, test_path=None, dev_path=None, config=None):
        if config is None:
            config = self.Config()

        learning_rate = config.learning_rate
        c = config.c
        rho = config.rho
        batch_size = config.batch_size
        max_iter = config.max_iter
        check_point = config.check_point
        save_iter = config.save_iter
        delay_step = config.delay_step
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
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self._create_feature_space(data, pos)

        convergence = False
        iter_count = 0
        tags = self.model.tags
        weight = self.model.weight
        dw = np.zeros(self.model.feature_size)
        change_set = set()
        b = 0
        global_step = 0
        extract_feature = self._extract_feature
        times = []
        while not convergence:
            dr.shuffle()
            data = dr.get_seg_data()
            pos = dr.get_pos_data()
            start = datetime.now()
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                for k in range(s_len):
                    features = np.array([extract_feature(sentence, k, now_tag) for now_tag in tags.values()])
                    wf = self._dot(features)  # (tag_size)
                    wf_lsm = logsumexp(wf)
                    change_set.update(features.reshape(-1))
                    dw[features] -= np.exp(wf - wf_lsm).reshape(-1, 1)
                    dw[features[tags[pos[i][k]]]] += 1

                    b += 1
                    if b == batch_size:
                        current_learning_rate = learning_rate * rho ** (global_step // delay_step)
                        global_step += 1
                        change_array = np.array(list(change_set))
                        weight[change_array] += current_learning_rate * dw[change_array]
                        weight[0] = 0
                        b = 0
                        dw[change_array] = 0
                        change_set.clear()

            iter_count += 1

            _, _, train_acc = self.evaluate(eval_reader=dr)
            print(f"iter: {iter_count} train accuracy: {train_acc :.5%}")
            if dev_reader is not None:
                _, _, dev_acc = self.evaluate(eval_reader=dev_reader)
                print(f"iter: {iter_count} dev   accuracy: {dev_acc :.5%}")
            if test_reader is not None:
                _, _, test_acc = self.evaluate(eval_reader=test_reader)
                print(f"iter: {iter_count} test  accuracy: {test_acc :.5%}")

            spend = datetime.now() - start
            times.append(spend)
            if iter_count >= max_iter:
                convergence = True
                avg_spend = sum(times, timedelta(0)) / len(times)
                print(f"iter: training average spend time: {avg_spend}s\n")
                if check_point:
                    self.save_model(check_point + 'check_point_finish.pickle')
            else:
                if c != 0:
                    weight *= (1 - c)
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(check_point + 'check_point_' + str(iter_count) + '.pickle')
                print(f"iter: {iter_count} spend time: {spend}s\n")

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def tag(self, s, index=None):
        assert self.model
        if index is None:
            return [self._tag(s, i) for i in range(len(s))]
        else:
            return self._tag(s, index)

    def _tag(self, s, index):
        extract_feature = self._extract_feature
        t = self.model.tags
        f = np.array([extract_feature(s, index, now_tag) for now_tag in t.values()])  # (tag_size, feature_size)
        scores = self._dot(f)
        return self.model.tags_backward[np.argmax(scores)]

    def _dot(self, feature_vector):
        return np.sum(self.model.weight[feature_vector], axis=-1)

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

    def _extract_feature(self, s, index, now_tag, new_id=False):
        wi = s[index]

        if index > 0:
            wim1 = s[index - 1]
        else:
            wim1 = "^^"

        s_len = len(s)
        w_len = len(wi)

        if index < s_len - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"
        feature_vector = [self._get_feature_id((2, now_tag, s[index]), new_id=new_id),
                          self._get_feature_id((3, now_tag, wim1), new_id=new_id),
                          self._get_feature_id((4, now_tag, wip1), new_id=new_id),
                          self._get_feature_id((5, now_tag, wi, wim1[-1]), new_id=new_id),
                          self._get_feature_id((6, now_tag, wi, wip1[0]), new_id=new_id),
                          self._get_feature_id((7, now_tag, wi[0]), new_id=new_id),
                          self._get_feature_id((8, now_tag, wi[-1]), new_id=new_id)]

        for k in range(1, w_len - 1):
            feature_vector += [self._get_feature_id((9, now_tag, wi[k]), new_id=new_id),
                               self._get_feature_id((10, now_tag, wi[0], wi[k]), new_id=new_id),
                               self._get_feature_id((11, now_tag, wi[-1], wi[k]), new_id=new_id)]

        if w_len == 1:
            feature_vector += [self._get_feature_id((12, now_tag, wi, wim1[-1], wip1[0]), new_id=new_id)]

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector += [self._get_feature_id((13, now_tag, wi[k], "__C0nsecut1ve?__"), new_id=new_id)]

        for k in range(1, min(5, w_len + 1)):
            feature_vector += [self._get_feature_id((14, now_tag, wi[:k]), new_id=new_id),
                               self._get_feature_id((15, now_tag, wi[-k:]), new_id=new_id)]

        return feature_vector

    def _create_feature_space(self, segs, tags):
        data_len = len(segs)
        t = self.model.tags
        t_b = self.model.tags_backward
        for i in range(data_len):
            sentence = segs[i]
            s_len = len(sentence)
            for k in range(s_len):
                tag = tags[i][k]
                if tag not in t:
                    t_id = len(t)
                    t[tag] = t_id
                    t_b[t_id] = tag
                self._extract_feature(sentence, k, t[tag], new_id=True)

        self.model.tag_size = len(t)
        self.model.feature_size = len(self.model.features)
        self.model.weight = np.zeros(self.model.feature_size)


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll',
                 dev_path='.\\data\\dev.conll',
                 # test_path='.\\bigdata\\test.conll',
                 config=Tagger.Config(learning_rate=0.5,  # data 0.5 is fine
                                      c=0,
                                      rho=1,
                                      delay_step=100000,
                                      max_iter=100,
                                      batch_size=50,
                                      check_point='.\\model\\',
                                      save_iter=5))
    tagger.save_model('.\\model\\data_model.pickle')
    tagger.load_model('.\\model\\data_model.pickle')
