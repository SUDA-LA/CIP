import pickle
from datetime import datetime
from datetime import timedelta
from .DataReader import DataReader
import numpy as np
from .TaggerBase import TaggerBase


class Tagger(TaggerBase):
    def __init__(self, model_path=None):
        super(Tagger, self).__init__(model_path=model_path)

    class Model:
        def __init__(self):
            self.weight = None
            self.v = None
            self.tags = {}
            self.tags_backward = {}
            self.features = {0: 0}
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
        weight = self.model.weight
        weight[0] = 0
        v = self.model.v
        t = self.model.tags

        convergence = False
        iter_count = 0
        extract_feature = self._extract_feature
        tagging = self.tag
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

                    if random_lr is None:
                        p_rate = 1
                        n_rate = 1
                    else:
                        p_rate = random_lr()
                        n_rate = random_lr()

                    for k in range(s_len):
                        gt_features_k = gt_features[k]
                        pred_features_k = pred_features[k]
                        update_set.update(gt_features_k)
                        update_set.update(pred_features_k)

                        weight_temp[gt_features_k] += p_rate
                        weight_temp[pred_features_k] -= n_rate
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
                    self.save_model(check_point + 'check_point_finish.pickle')
            else:
                if check_point and (iter_count % save_iter) == 0:
                    self.save_model(check_point + 'check_point_' + str(iter_count) + '.pickle')
                print(f"iter: {iter_count} spend time: {spend}s\n")

    def _tag(self, s, index, averaged_perceptron=False):
        extract_feature = self._extract_feature
        t = self.model.tags
        f = np.array([extract_feature(s, index, now_tag) for now_tag in t.values()])  # (tag_size, feature_size)
        score = self._dot(f, averaged_perceptron=averaged_perceptron)                 # (tag_size)
        return self.model.tags_backward[np.argmax(score)]

    def _dot(self, feature_vector, averaged_perceptron=False):
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
        self.model.v = np.zeros(self.model.feature_size)


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll',
                 dev_path='.\\data\\dev.conll',
                 # test_path='.\\bigdata\\test.conll',
                 config=tagger.Config(0, 100, '.\\model\\', 5,
                                      averaged_perceptron=True,
                                      random_lr=lambda: 0.8 + 0.4 * np.random.random()))
                                      # random_lr=lambda: np.random.normal(1, 0.2)))
    tagger.save_model('.\\model\\data_model.pickle')
    tagger.load_model('.\\model\\data_model.pickle')
