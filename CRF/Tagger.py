import pickle
from DataReader import DataReader
import numpy as np
from scipy.special import logsumexp
from datetime import datetime
from datetime import timedelta


class Tagger:
    def __init__(self, model_path=None):
        self.model_name = 'CRF Tagger'
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
            tag = self.tag(val)
            right += len([index for index, v in enumerate(tag) if v == p[i][index]])
            word_count += len(tag)

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
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_len = len(data)
        self.model = self.Model()
        self._create_feature_space(data, pos)
        weight = self.model.weight
        t = self.model.tags

        convergence = False
        iter_count = 0
        update_set = set()
        g = np.zeros(self.model.feature_size)
        global_step = 0
        b = 0
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
                tf = self.model.tags_feature  # (now, pre)
                gt_pos = [int(t[p]) for p in pos[i]]
                scores, features = self._get_scores(sentence, with_features=True)
                alpha, beta = self._get_alpha_beta(scores)
                _, pred_pos_name = self._viterbi_decode(scores)
                z = np.average([alpha[s_len + 1][1], beta[0][0]])
                for k in range(s_len + 1):
                    gt_k = gt_pos[k] if k < s_len else 1
                    gt_k_1 = gt_pos[k - 1] if k > 0 else 0
                    features_k = features[k]

                    update_set.update(features_k.reshape(-1))
                    # 计算dlogZ/dw
                    update = np.exp(alpha[k].reshape((1, -1)) + scores[k] + beta[k + 1].reshape((-1, 1)) - z)
                    # (1, pre) + (now, pre) + (now, 1) - () -> (now, pre)
                    # 0维向量无法转置

                    # 计算更新梯度
                    g[tf[gt_k, gt_k_1]] += 1
                    g[features_k[gt_k]] += 1

                    g[tf] -= update  # (now, pre)
                    g[features_k] -= np.sum(update, axis=1).reshape(-1, 1)
                    # (now_tag, list(feature)) - (now , 1)

                    b += 1

                    if b >= batch_size:
                        current_learning_rate = learning_rate * rho ** (global_step // delay_step)
                        global_step += 1
                        update_set.update(tf.reshape(-1))
                        update_set -= {0}
                        update_list = np.array(list(update_set))
                        update_set.clear()
                        weight[update_list] += current_learning_rate * g[update_list]

                        g[update_list] = 0
                        b = 0

                word_count += s_len
                wrong += len([False for i, tag in enumerate(pos[i]) if tag != pred_pos_name[i]])

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
            pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _viterbi_decode(self, scores):
        s_len = len(scores) - 1
        backwards = np.zeros((s_len + 1, self.model.tag_size))
        score = np.full((1, self.model.tag_size), -np.inf)
        score[0][0] = 0
        for i in range(s_len + 1):
            score_i = score + scores[i]  # （1, pre) + (now, pre)
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

    def _get_scores(self, s, with_features=False):
        assert self.model
        s_len = len(s)
        t = self.model.tags
        emit_scores = np.zeros((s_len + 1, self.model.tag_size))
        extract_feature = self._extract_feature
        end_feature = self._end_feature
        features = []
        for i in range(s_len):
            f = np.array([extract_feature(s, i, now_tag) for now_tag in t.values()])  # (now_tag, feature)
            features += [f]
            emit_scores[i] = self._dot(f)    # (s_len + 1, now)
        f = np.array([end_feature(s, now_tag) for now_tag in t.values()])  # (now_tag, feature)
        features += [f]
        emit_scores[-1] = self._dot(f)
        tf = self.model.tags_feature  # (now, pre, feature)
        trans_scores = self._dot(tf[:, :, np.newaxis])  # (now, pre)

        # 开始的时候end_feature直接给的None (这个没什么毛病)
        scores = emit_scores[:, :, np.newaxis] + trans_scores[np.newaxis, :, :]
        # (s_len + 1, now, 1) + (1, now, pre)
        if with_features:
            return scores, features
        else:
            return scores

    def _get_alpha_beta(self, scores):
        s_len = len(scores) - 1
        # 这里的alpha和beta准确地说是log(alpha)和log(beta)
        alpha = np.full((s_len + 2, self.model.tag_size), -np.inf)
        alpha[0][0] = 0
        beta = np.full((s_len + 2, self.model.tag_size), -np.inf)
        beta[s_len + 1][1] = 0
        scores_t = scores.transpose((0, 2, 1))
        for k in range(s_len + 1):
            alpha[k + 1] = logsumexp(scores[k] + alpha[k].reshape((1, -1)), axis=1)
            # (now, pre) + (1, pre) -> (now)
            beta[s_len - k] = logsumexp(scores_t[s_len - k] + beta[s_len - k + 1].reshape((1, -1)), axis=1)
            # (pre, now) + (1, now) -> (pre)
            # 开始的时候beta中的scores没有转置 (×)
        return alpha, beta

    def tag(self, s):
        scores = self._get_scores(s)
        _, tags = self._viterbi_decode(scores)
        return tags

    def _dot(self, feature_vector):
        if len(feature_vector) == 0:
            return 0
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

    def _end_feature(self, s, now_tag, new_id=False):
        if now_tag != 1:
            return [0, 0, 0, 0, 0, 0]
        else:
            wi = "$$"
            wim1 = s[-1]

            feature_vector = [self._get_feature_id((2, now_tag, wi), new_id=new_id),
                              self._get_feature_id((3, now_tag, wim1), new_id=new_id),
                              self._get_feature_id((5, now_tag, wi, wim1[-1]), new_id=new_id),
                              self._get_feature_id((7, now_tag, wi[0]), new_id=new_id),
                              self._get_feature_id((8, now_tag, wi[-1]), new_id=new_id),
                              self._get_feature_id((13, now_tag, wi[0]), new_id=new_id)]

            return feature_vector

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
        get_feature_id = self._get_feature_id
        feature_vector = [get_feature_id((2, now_tag, s[index]), new_id=new_id),
                          get_feature_id((3, now_tag, wim1), new_id=new_id),
                          get_feature_id((4, now_tag, wip1), new_id=new_id),
                          get_feature_id((5, now_tag, wi, wim1[-1]), new_id=new_id),
                          get_feature_id((6, now_tag, wi, wip1[0]), new_id=new_id),
                          get_feature_id((7, now_tag, wi[0]), new_id=new_id),
                          get_feature_id((8, now_tag, wi[-1]), new_id=new_id)]

        w_len = len(wi)

        for k in range(1, w_len - 1):
            feature_vector += [get_feature_id((9, now_tag, wi[k]), new_id=new_id),
                               get_feature_id((10, now_tag, wi[0], wi[k]), new_id=new_id),
                               get_feature_id((11, now_tag, wi[-1], wi[k]), new_id=new_id)]

        if w_len == 1:
            feature_vector += [get_feature_id((12, now_tag, wi, wim1[-1], wip1[0]), new_id=new_id)]

        for k in range(w_len - 1):
            if wi[k] == wi[k + 1]:
                feature_vector += [get_feature_id((13, now_tag, wi[k], "__C0nsecut1ve?__"), new_id=new_id)]

        for k in range(1, min(5, w_len + 1)):
            feature_vector += [get_feature_id((14, now_tag, wi[:k]), new_id=new_id),
                               get_feature_id((15, now_tag, wi[-k:]), new_id=new_id)]

        return feature_vector

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
        tags_feature = self._tags_feature
        self.model.tags_feature = np.array([
            [tags_feature(pred_tag, now_tag) for pred_tag in t.values()]
            for now_tag in t.values()
        ])


if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    # tagger.train('.\\bigdata\\train.conll', tagger.Config(0, 20, '.\\model\\', 1))
    tagger.train('.\\data\\train.conll',
                 dev_path='.\\data\\dev.conll',
                 #test_path='.\\bigdata\\test.conll',
                 config=Tagger.Config(learning_rate=0.4,  # data 0.4 is fine big data use < 0.1
                                      c=0,
                                      rho=1,
                                      delay_step=100000,
                                      max_iter=50,
                                      batch_size=50,
                                      check_point='.\\model\\',
                                      save_iter=5))
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')