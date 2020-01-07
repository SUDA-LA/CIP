import datetime
import numpy as np
import random
from scipy.misc import logsumexp
from collections import defaultdict
from config import config


class dataset(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        f = open(filename, encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.sentences_num = len(self.sentences)
        self.word_num = word_num

        print('%s:共%d个句子,共%d个词。' % (filename, self.sentences_num, self.word_num))
        f.close()

    def shuffle(self):
        temp = [(s, t) for s, t in zip(self.sentences, self.tags)]
        random.shuffle(temp)
        self.sentences = []
        self.tags = []
        for s, t in temp:
            self.sentences.append(s)
            self.tags.append(t)


class CRF(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file != None else None
        self.dev_data = dataset(dev_data_file) if dev_data_file != None else None
        self.test_data = dataset(test_data_file) if test_data_file != None else None
        self.features = {}
        self.weights = []
        self.tag2id = {}
        self.tags = []
        self.EOS = 'EOS'
        self.BOS = 'BOS'

    def create_bigram_feature(self, pre_tag, cur_tag):
        return ['01:' + cur_tag + '*' + pre_tag]

    def create_unigram_feature(self, sentence, position, cur_tag):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_tag + '*' + cur_word)
        template.append('03:' + cur_tag + '*' + last_word)
        template.append('04:' + cur_tag + '*' + next_word)
        template.append('05:' + cur_tag + '*' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_tag + '*' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_tag + '*' + cur_word_first_char)
        template.append('08:' + cur_tag + '*' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + cur_tag + '*' + sentence[position][i])
            template.append('10:' + cur_tag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + cur_tag + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + cur_tag + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + cur_tag + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_tag + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + cur_tag + '*' + sentence[position][0:i + 1])
            template.append('15:' + cur_tag + '*' + sentence[position][-(i + 1)::])
        return template

    def create_feature_template(self, sentence, position, pre_tag, cur_tag):
        template = []
        template.extend(self.create_bigram_feature(pre_tag, cur_tag))
        template.extend(self.create_unigram_feature(sentence, position, cur_tag))
        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                if j == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = tags[j - 1]
                template = self.create_feature_template(sentence, j, pre_tag, tags[j])
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
        self.tags = sorted(self.tags)
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.weights = np.zeros(len(self.features))
        self.g = defaultdict(float)
        self.bigram_features = [
            [self.create_bigram_feature(prev_tag, tag) for prev_tag in self.tags]
            for tag in self.tags
        ]
        self.bigram_scores = np.zeros((len(self.tags), len(self.tags)))
        print("the total number of features is %d" % (len(self.features)))

    def score(self, feature):
        scores = [self.weights[self.features[f]] for f in feature if f in self.features]
        return sum(scores)

    def predict(self, sentence):
        states = len(sentence)
        type = len(self.tag2id)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type), dtype='int')

        for j in range(type):
            feature = self.create_feature_template(sentence, 0, self.BOS, self.tags[j])
            max_score[0][j] = self.score(feature)
            paths[0][j] = -1

        # 动态规划
        for i in range(1, states):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence, i, tag)) for tag in self.tags])
            scores = self.bigram_scores + unigram_scores[:, None] + max_score[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            max_score[i] = np.max(scores, axis=1)
            # for j in range(type):
            #     unigram_scores = self.score(self.create_unigram_feature(sentence, i, self.tags[j]))
            #     scores = unigram_scores + np.array(self.bigram_scores[j])
            #     max_score[i][j] = max(scores + max_score[i - 1])
            #     paths[i][j] = np.argmax(scores + max_score[i - 1])

        gold_path = []
        cur_state = states - 1
        step = np.argmax(max_score[cur_state])
        gold_path.insert(0, self.tags[step])
        while True:
            step = int(paths[cur_state][step])
            if step == -1:
                break
            gold_path.insert(0, self.tags[step])
            cur_state -= 1
        return gold_path

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            predict = self.predict(sentence)
            for j in range(len(tags)):
                if tags[j] == predict[j]:
                    correct_num += 1

        return correct_num, total_num, correct_num / total_num

    def forward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tags)))
        path_scores[0] = [self.score(self.create_feature_template(sentence, 0, self.BOS, tag))
                          for tag in self.tags]
        for i in range(1, len(sentence)):
            unigram_scores = np.array([self.score(self.create_unigram_feature(sentence, i, tag)) for tag in self.tags])
            scores = self.bigram_scores + unigram_scores[:, None]
            path_scores[i] = logsumexp(path_scores[i - 1] + scores, axis=1)

        return path_scores

    def backward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tags)))

        for i in reversed(range(len(sentence) - 1)):
            unigram_scores = np.array(
                [self.score(self.create_unigram_feature(sentence, i + 1, tag)) for tag in self.tags])
            scores = self.bigram_scores.T + unigram_scores
            path_scores[i] = logsumexp(path_scores[i + 1] + scores, axis=1)
        return path_scores

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
            else:
                pre_tag = tags[i - 1]
            cur_tag = tags[i]
            feature = self.create_feature_template(sentence, i, pre_tag, cur_tag)
            for f in feature:
                if f in self.features:
                    self.g[self.features[f]] += 1

        forward_scores = self.forward(sentence)
        backward_scores = self.backward(sentence)
        dinominator = logsumexp(forward_scores[-1])

        for i, tag in enumerate(self.tags):
            features = self.create_feature_template(sentence, 0, self.BOS, tag)
            features_id = (self.features[f] for f in features if f in self.features)
            p = np.exp(self.score(features) + backward_scores[0, i] - dinominator)
            for id in features_id:
                self.g[id] -= p

        for i in range(1, len(sentence)):
            for j, tag in enumerate(self.tags):
                unigram_features = self.create_unigram_feature(sentence, i, tag)
                unigram_features_id = [self.features[f] for f in unigram_features if f in self.features]
                scores = self.bigram_scores[j] + self.score(unigram_features)
                probs = np.exp(scores + forward_scores[i - 1] + backward_scores[i, j] - dinominator)

                for bigram_feature, p in zip(self.bigram_features[j], probs):
                    bigram_feature_id = [self.features[f]
                                         for f in bigram_feature if f in self.features]
                    for fi in bigram_feature_id + unigram_features_id:
                        self.g[fi] -= p

    def SGD_train(self, iteration=100, batchsize=1, shuffle=False, regulization=False, step_opt=False, eta=0.5,
                  C=0.0001, exitor=10):
        max_dev_precision = 0
        counter = 0
        global_step = 1
        decay_steps = len(self.train_data.sentences) / batchsize
        decay_rate = 0.96
        learn_rate = eta
        print('eta=%f' % eta)
        if regulization:
            print('add regulization...C=%f' % (C), flush=True)
        if step_opt:
            print('add step optimization', flush=True)
        for iter in range(iteration):
            b = 0
            starttime = datetime.datetime.now()
            print('iterator: %d' % (iter), flush=True)
            if shuffle:
                print('shuffle the train data...', flush=True)
                self.train_data.shuffle()
            for i in range(len(self.train_data.sentences)):
                b += 1
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                self.update_gradient(sentence, tags)
                if b == batchsize:
                    if regulization:
                        self.weights *= (1 - C * eta)
                    for id, value in self.g.items():
                        self.weights[id] += value * learn_rate
                    if step_opt:
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                    global_step += 1
                    self.g = defaultdict(float)
                    self.bigram_scores = np.array([
                        [self.score(f) for f in bigram_features]
                        for bigram_features in self.bigram_features
                    ])
                    b = 0

            if b > 0:
                if regulization:
                    self.weights *= (1 - C * eta)
                for id, value in self.g.items():
                    self.weights[id] += value * learn_rate
                if step_opt:
                    learn_rate = eta * decay_rate ** (global_step / decay_steps)
                global_step += 1
                self.g = defaultdict(float)
                self.bigram_scores = np.array([
                    [self.score(f) for f in bigram_features]
                    for bigram_features in self.bigram_features
                ])
                b = 0

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision), flush=True)
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)

            if self.test_data != None:
                test_correct_num, test_num, test_precision = self.evaluate(self.test_data)
                print('\t' + 'test准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision), flush=True)

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1

            endtime = datetime.datetime.now()
            print("\titeration executing time is " + str((endtime - starttime)) + " s", flush=True)
            if counter >= exitor:
                break
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)


if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    iterator = config['iterator']
    batchsize = config['batchsize']
    shuffle = config['shuffle']
    regulization = config['regulization']
    step_opt = config['step_opt']
    C = config['C']
    eta = config['eta']
    exitor = config['exitor']

    starttime = datetime.datetime.now()
    crf = CRF(train_data_file, dev_data_file, test_data_file)
    crf.create_feature_space()
    print(crf.tag2id)
    crf.SGD_train(iterator, batchsize, shuffle, regulization, step_opt, eta, C, exitor)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
