import pickle
from DataReader import DataReader
import numpy as np
from datetime import datetime
from datetime import timedelta


class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'Optimized Global Linear Tagger'
        self.model = None
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
        def __init__(self, stop_threshold=0,
                     max_iter=30,
                     check_point=None,
                     save_iter=5,
                     averaged_perceptron=False,
                     random_lr=None,
                     evaluate_mode=False):
            self.stop_threshold = stop_threshold
            self.max_iter = max_iter
            self.check_point = check_point
            self.save_iter = save_iter
            self.averaged_perceptron = averaged_perceptron
            self.random_lr = random_lr
            self.evaluate_mode = evaluate_mode

    def load_model(self, model_path):
        self.model = None
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

    def train(self, data_path, test_path=None, dev_path=None, config=None):
        # decode config
        if config is None:
            config = self.Config()

        stop_threshold = config.stop_threshold
        max_iter = config.max_iter
        check_point = config.check_point
        save_iter = config.save_iter
        averaged_perceptron = config.averaged_perceptron
        random_lr = config.random_lr
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

        # prepare data
        self.model = self.Model()
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self._create_feature_space(data, pos)
        weight = self.model.weight
        tag_v = self.model.tag_v
        v = self.model.v
        t = self.model.tags

        # init
        convergence = False
        iter_count = 0
        global_step = 0
        times = []
        tag_weight = self.model.tag_weight
        update_set = set()
        update_step = np.zeros(self.model.feature_size)
        weight_temp = np.zeros((self.model.tag_size, self.model.feature_size))
        while not convergence:
            wrong = 0
            word_count = 0
            dr.shuffle()
            data = dr.get_seg_data()
            pos = dr.get_pos_data()
            start = datetime.now()
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                features = [self._extract_feature(sentence, k) for k in range(s_len)]
                gt_pos = [int(t[p]) for p in pos[i]]
                pred_pos = [int(t[p]) for p in self.tag(sentence)]
                if gt_pos != pred_pos:

                    if random_lr is None:
                        p_rate = 1
                        n_rate = 1
                    else:
                        p_rate = random_lr()
                        n_rate = random_lr()

                    for k in range(s_len):
                        gt_k = gt_pos[k]
                        gt_k_1 = gt_pos[k - 1] if k > 0 else 0
                        pred_k = pred_pos[k]
                        pred_k_1 = pred_pos[k - 1] if k > 0 else 0
                        feature_k = features[k]

                        update_set.update(feature_k)
                        tag_weight[gt_k, gt_k_1] += p_rate
                        weight_temp[gt_k, feature_k] += p_rate
                        tag_weight[pred_k, pred_k_1] -= n_rate
                        weight_temp[pred_k, feature_k] -= n_rate

                    tag_v += tag_weight
                    update_list = np.array(list(update_set))
                    update_set.clear()
                    v[:, update_list] += (global_step - update_step[update_list]).reshape((1, -1)) * weight[:, update_list]
                    # ((1) - (update_size)) -> (1, update_size) * (tag_size, update_size)
                    update_step[update_list] = global_step
                    weight[:, update_list] += weight_temp[:, update_list]
                    weight_temp[:, update_list] = 0
                    global_step += 1

                word_count += s_len
                wrong += len([False for i, tag in enumerate(gt_pos) if tag != pred_pos[i]])

            v += (global_step - update_step) * weight
            update_step[:] = global_step

            iter_count += 1

            _, _, train_acc = self.evaluate(eval_reader=dr, averaged_perceptron=averaged_perceptron)
            print(f"iter: {iter_count} train accuracy: {train_acc :.5%}")
            if dev_reader is not None:
                _, _, dev_acc = self.evaluate(eval_reader=dev_reader, averaged_perceptron=averaged_perceptron)
                print(f"iter: {iter_count} dev   accuracy: {dev_acc :.5%}")
            if test_reader is not None:
                _, _, test_acc = self.evaluate(eval_reader=test_reader, averaged_perceptron=averaged_perceptron)
                print(f"iter: {iter_count} test  accuracy: {test_acc :.5%}")
            loss = wrong / word_count
            spend = datetime.now() - start
            times.append(spend)
            if loss <= stop_threshold or iter_count >= max_iter:
                convergence = True
                avg_spend = sum(times, timedelta(0)) / len(times)
                print(f"iter: training average spend time: {avg_spend}s\n")
                if check_point:
                    self.save_model(check_point + 'check_point_finish.pickle')
            else:
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(check_point + 'check_point_' + str(iter_count) + '.pickle')
                print(f"iter: {iter_count} spend time: {spend}s\n")

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def tag(self, s, averaged_perceptron=False):
        assert self.model
        s_len = len(s)
        pi = np.full((s_len + 1, self.model.tag_size), -np.inf)
        bt = np.zeros((s_len, self.model.tag_size))
        pi[0][0] = 0
        if averaged_perceptron:
            tw = self.model.tag_v
        else:
            tw = self.model.tag_weight
        for i in range(s_len):
            f = self._extract_feature(s, i)
            scores = self._dot(f, averaged_perceptron=averaged_perceptron).reshape((-1, 1)) + tw + pi[i]
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

    def _dot(self, feature_vector, averaged_perceptron=False):
        if len(feature_vector) == 0:
            return 0
        else:
            if averaged_perceptron:
                return np.sum(self.model.v[:, feature_vector], axis=-1)
            else:
                return np.sum(self.model.weight[:, feature_vector], axis=-1)

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

        s_len = len(s)

        if index < s_len - 1:
            wip1 = s[index + 1]
        else:
            wip1 = "$$"
        feature_vector = [self._get_feature_id((2, s[index]), new_id=new_id),
                          self._get_feature_id((3, wim1), new_id=new_id),
                          self._get_feature_id((4, wip1), new_id=new_id),
                          self._get_feature_id((5, wi, wim1[-1]), new_id=new_id),
                          self._get_feature_id((6, wi, wip1[0]), new_id=new_id),
                          self._get_feature_id((7, wi[0]), new_id=new_id),
                          self._get_feature_id((8, wi[-1]), new_id=new_id),
                          self._get_feature_id((16, round((index + 0.5) * 10.0 / s_len)), new_id=new_id)]

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector += [self._get_feature_id((9, wi[k]), new_id=new_id),
                               self._get_feature_id((10, wi[0], wi[k]), new_id=new_id),
                               self._get_feature_id((11, wi[-1], wi[k]), new_id=new_id)]
            if wi[k] == wi[k + 1]:
                feature_vector += [self._get_feature_id((13, wi[k], "__C0nsecut1ve?__"), new_id=new_id)]

        if w_len == 1:
            feature_vector += [self._get_feature_id((12, wi, wim1[-1], wip1[0]), new_id=new_id)]
        else:
            if wi[0] == wi[1]:
                feature_vector += [self._get_feature_id((13, wi[0], "__C0nsecut1ve?__"), new_id=new_id)]

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
    tagger.train('.\\data\\train.conll',
                 dev_path='.\\data\\dev.conll',
                 # test_path='.\\bigdata\\test.conll',
                 config=tagger.Config(0, 30, '.\\model\\', 5, random_lr=lambda: np.random.normal(1, 0.2)))
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')
