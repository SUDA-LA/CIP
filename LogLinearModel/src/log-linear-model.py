import datetime
import numpy as np
import random
from collections import defaultdict
from scipy.misc import logsumexp
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
        while (True):
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

        print('%s:共%d个句子,共%d个词。' % (filename, self.sentences_num, self.word_num), flush=True)
        f.close()

    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                data.append((self.sentences[i], j, self.tags[i][j]))
        return data


class loglinear_model(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file != None else None
        self.dev_data = dataset(dev_data_file) if dev_data_file != None else None
        self.test_data = dataset(test_data_file) if test_data_file != None else None
        self.weights = []
        self.features = {}
        self.g = []
        self.tag_list = []
        self.tag_dict = {}
        return

    def create_feature_template(self, sentence, tag, position):
        template = []
        cur_word = sentence[position]
        cur_tag = tag
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

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                template = self.create_feature_template(sentence, tags[j], j)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)
        self.tag_list = sorted(self.tag_list)
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.weights = np.zeros(len(self.features))
        self.g = defaultdict(float)
        print("the total number of features is %d" % (len(self.weights)), flush=True)

    def dot(self, feature):
        score = 0.0
        for f in feature:
            if f in self.features:
                score += self.weights[self.features[f]]
        return score

    def predict(self, sentence, position):
        fvectors = [self.create_feature_template(sentence, tag, position) for tag in self.tag_list]
        scores = [self.dot(template) for template in fvectors]
        return self.tag_list[np.argmax(scores)]

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            for j in range(len(sentence)):
                predict_tag = self.predict(sentence, j)
                if predict_tag == tags[j]:
                    correct_num += 1

        return (correct_num, total_num, correct_num / total_num)

    def SGD_train(self, iteration, batch_size=50, shuffle=False, regulization=False, step_opt=False, eta=0.5, C=0.0001,
                  exitor=10):
        b = 0
        max_dev_precision = 0
        counter = 0
        global_step = 1
        decay_steps = 100000
        decay_rate = 0.96
        learn_rate = eta

        data = self.train_data.split()
        print('eta=%f' % eta)
        if regulization:
            print('add regulization...C=%f' % (C), flush=True)
        if step_opt:
            print('add step optimization', flush=True)
        for iter in range(iteration):
            print('iterator: %d' % (iter), flush=True)
            starttime = datetime.datetime.now()
            if shuffle:
                print('shuffle the train data...', flush=True)
                random.shuffle(data)
            for i in range(len(data)):
                b += 1
                sentence = data[i][0]
                j = data[i][1]
                gold_tag = data[i][2]
                gold_feature = self.create_feature_template(sentence, gold_tag, j)
                for f in gold_feature:
                    if f in self.features:
                        self.g[self.features[f]] += 1

                feature_list = [self.create_feature_template(sentence, tag, j) for tag in self.tag_list]
                scores = [self.dot(template) for template in feature_list]
                s = logsumexp(scores)
                prob_list = np.array(scores) - s

                for template, p in zip(feature_list, prob_list):
                    for f in template:
                        if f in self.features:
                            self.g[self.features[f]] -= np.exp(p)

                if b == batch_size:
                    if regulization:
                        self.weights *= (1 - C * learn_rate)
                    for id, value in self.g.items():
                        self.weights[id] += eta * value
                    if step_opt:
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                    b = 0
                    global_step += 1
                    self.g = defaultdict(float)

            if b > 0:
                if regulization:
                    self.weights *= (1 - C * learn_rate)
                for id, value in self.g.items():
                    self.weights[id] += eta * value
                if step_opt:
                    learn_rate = eta * decay_rate ** (global_step / decay_steps)
                b = 0
                global_step += 1
                self.g = defaultdict(float)

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
            print("\titeration executing time is " + str((endtime - starttime).seconds) + " s", flush=True)
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
    exitor = config['exitor']
    regulization = config['regulization']
    step_opt = config['step_opt']
    C = config['C']
    eta = config['eta']

    starttime = datetime.datetime.now()
    lm = loglinear_model(train_data_file, dev_data_file, test_data_file)
    lm.create_feature_space()
    lm.SGD_train(iterator, batchsize, shuffle, regulization, step_opt, eta, C, exitor)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
