from DataReader import DataReader
from datetime import datetime
from datetime import timedelta
import numpy as np
from TaggerBase import TaggerBase
import os

class Tagger(TaggerBase):
    def __init__(self, model_path=None):
        super(Tagger, self).__init__(model_path=model_path)
        self.model_name = 'Optimized Linear Tagger'

    class Model:
        def __init__(self):
            self.weight = None
            self.v = None
            self.tags = {}
            self.tags_backward = {}
            self.features = {}
            self.tag_size = 0
            self.feature_size = 0

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

        convergence = False
        iter_count = 0
        global_step = 0
        update_set = set()
        update_step = np.zeros(self.model.feature_size)
        weight_temp = np.zeros((self.model.tag_size, self.model.feature_size))
        times = []
        while not convergence:
            wrong = 0
            word_count = 0
            start = datetime.now()
            for i in range(data_len):
                sentence = data[i]
                s_len = len(sentence)
                features = [self._extract_feature(sentence, k) for k in range(s_len)]
                gt_pos = [int(t[p]) for p in pos[i]]
                pred_pos = [int(t[p]) for p in self.tag(sentence)]
                if gt_pos != pred_pos:

                    if random_lr:
                        p_rate = min_lr + ((max_lr - min_lr) * np.random.random())
                        n_rate = min_lr + ((max_lr - min_lr) * np.random.random())
                    else:
                        p_rate = 1
                        n_rate = 1

                    for k in range(s_len):
                        gt_k = gt_pos[k]
                        pred_k = pred_pos[k]
                        feature_k = features[k]

                        update_set.update(feature_k)
                        weight_temp[gt_k, feature_k] += p_rate
                        weight_temp[pred_k, feature_k] -= n_rate

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
                    self.save_model(os.path.join(check_point,'check_point_finish.pickle'))
            else:
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(os.path.join(check_point, f'check_point_{iter_count}.pickle'))
                print(f"iter: {iter_count} spend time: {spend}s\n")

    def tag(self, s, index=None, averaged_perceptron=False):
        assert self.model
        if index is None:
            return [self._tag(s, i, averaged_perceptron=averaged_perceptron) for i in range(len(s))]
        else:
            return self._tag(s, index, averaged_perceptron=averaged_perceptron)

    def _tag(self, s, index, averaged_perceptron=False):
        f = self._extract_feature(s, index)  # np.array
        score = self._dot(f, averaged_perceptron=averaged_perceptron)                 # (tag_size)
        return self.model.tags_backward[np.argmax(score)]

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
        self.model.weight = np.zeros((self.model.tag_size, self.model.feature_size))
        self.model.v = np.zeros((self.model.tag_size, self.model.feature_size))
