import pickle
from DataReader import DataReader
import numpy as np
from datetime import datetime
from datetime import timedelta
import os

class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'Global Linear Tagger'
        self.model = None
        if model_path:
            self.load_model(model_path)

    class Model:
        def __init__(self):
            self.weight = None
            self.v = None
            self.tags = {}
            self.tags_backward = {}
            self.features = {0: 0}
            self.tags_feature = None
            self.tag_size = 0
            self.feature_size = 0

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
        if config is None:
            config = self.Config()

        stop_threshold = config.stop_threshold
        max_iter = config.max_iter
        check_point = config.check_point
        save_iter = config.save_iter
        averaged_perceptron = config.averaged_perceptron
        random_lr = config.random_lr
        max_lr = config.max_lr
        min_lr = config.min_lr

        dr = DataReader(data_path, random_seed=config.seed)
        print(f"Set the seed for built-in generating random numbers to {config.seed}")
        np.random.seed(config.seed)
        print(f"Set the seed for numpy generating random numbers to {config.seed}")

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
        weight = self.model.weight
        weight[0] = 0
        v = self.model.v
        t = self.model.tags
        extract_feature = self._extract_feature
        tags_feature = self._tags_feature
        tagging = self.tag

        convergence = False
        iter_count = 0
        global_step = 0
        update_set = set()
        update_step = np.zeros(self.model.feature_size)
        weight_temp = np.zeros(self.model.feature_size)
        times = []
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
                gt_pos = [int(t[p]) for p in pos[i]]
                pred_pos = [int(t[p]) for p in tagging(sentence)]
                if gt_pos != pred_pos:
                    gt_features = [
                        extract_feature(sentence, k, gt_pos[k])
                        for k in range(s_len)
                    ]
                    pred_features = [
                        extract_feature(sentence, k, pred_pos[k])
                        for k in range(s_len)
                    ]
                    gt_tag_feature = np.array([
                        tags_feature(gt_pos[k - 1] if k > 0 else 0, gt_pos[k])
                        for k in range(s_len)
                    ])
                    pred_tag_feature = np.array([
                        tags_feature(pred_pos[k - 1] if k > 0 else 0, pred_pos[k])
                        for k in range(s_len)
                    ])

                    if random_lr:
                        p_rate = min_lr + ((max_lr - min_lr) * np.random.random())
                        n_rate = min_lr + ((max_lr - min_lr) * np.random.random())
                    else:
                        p_rate = 1
                        n_rate = 1

                    update_set.update(gt_tag_feature.reshape(-1))
                    update_set.update(pred_tag_feature.reshape(-1))

                    for k in range(s_len):
                        gt_features_k = gt_features[k]
                        pred_features_k = pred_features[k]
                        update_set.update(gt_features_k)
                        update_set.update(pred_features_k)

                        weight_temp[gt_features_k] += p_rate
                        weight_temp[gt_tag_feature[k]] += p_rate
                        weight_temp[pred_features_k] -= n_rate
                        weight_temp[pred_tag_feature[k]] -= n_rate
                        # weight[0] = 0  # 原本没有将Padding置零，导致严重过拟合

                    update_set -= {0}
                    update_list = np.array(list(update_set))  # (update_size)
                    update_set.clear()
                    v[update_list] += (global_step - update_step[update_list]) * weight[update_list]
                    update_step[update_list] = global_step
                    weight[update_list] += weight_temp[update_list]
                    weight_temp[update_list] = 0
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
                    self.save_model(os.path.join(check_point,'check_point_finish.pickle'))
            else:
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(os.path.join(check_point, f'check_point_{iter_count}.pickle'))
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
        t = self.model.tags
        extract_feature = self._extract_feature
        #tags_feature = self._tags_feature
        dot = self._dot
        argmax = np.argmax
        max = np.max
        tf = self.model.tags_feature
        # tf = np.array([
        #     [tags_feature(pred_tag, now_tag) for pred_tag in t.values()]
        #     for now_tag in t.values()
        # ])
        for i in range(s_len):
            f = np.array([extract_feature(s, i, now_tag) for now_tag in t.values()])  # (now_tag, feature)
              # (pred_tag, now_tag, 1)

            scores = dot(f, averaged_perceptron=averaged_perceptron).reshape((-1, 1)) + \
                     dot(tf[:, :, np.newaxis], averaged_perceptron=averaged_perceptron) + \
                     pi[i].reshape((1, -1))
            # (now_tag, 1) + (now_tag, pre_tag) + (1, pre_tag)
            bt[i] = argmax(scores, axis=1)
            pi[i + 1] = max(scores, axis=1)
        tags = [''] * s_len
        pre_tag = argmax(pi[-1])
        tags_backward = self.model.tags_backward
        for i in range(s_len):
            index = s_len - i - 1
            now_tag = int(pre_tag)
            tags[index] = tags_backward[now_tag]
            pre_tag = bt[index][now_tag]
        return tags

    def _dot(self, feature_vector, averaged_perceptron=False):
        if len(feature_vector) == 0:
            return 0
        else:
            if averaged_perceptron:
                return np.sum(self.model.v[feature_vector], axis=-1)
            else:
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

    def _tags_feature(self, pred_tag, now_tag, new_id=False):
        return self._get_feature_id((1, now_tag, pred_tag), new_id=new_id)

    def _extract_feature(self, s, index, now_tag, new_id=False):
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
        feature_vector = [self._get_feature_id((2, now_tag, s[index]), new_id=new_id),
                          self._get_feature_id((3, now_tag, wim1), new_id=new_id),
                          self._get_feature_id((4, now_tag, wip1), new_id=new_id),
                          self._get_feature_id((5, now_tag, wi, wim1[-1]), new_id=new_id),
                          self._get_feature_id((6, now_tag, wi, wip1[0]), new_id=new_id),
                          self._get_feature_id((7, now_tag, wi[0]), new_id=new_id),
                          self._get_feature_id((8, now_tag, wi[-1]), new_id=new_id)] #,
        # self._get_feature_id((16, now_tag, round((index + 0.5) * 10.0 / s_len)), new_id=new_id)]

        w_len = len(wi)

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
        t["START"] = 0
        t_b[0] = "START"
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
                self._tags_feature(t[tags[i][k - 1]] if k > 0 else 0, t[tag], new_id=True)
        self.model.tag_size = len(t)
        self.model.feature_size = len(self.model.features)
        print(f'feature size: {self.model.feature_size}')
        self.model.weight = np.zeros(self.model.feature_size)
        self.model.v = np.zeros(self.model.feature_size)
        tags_feature = self._tags_feature
        self.model.tags_feature = np.array([
            [tags_feature(pred_tag, now_tag) for pred_tag in t.values()]
            for now_tag in t.values()
        ])
