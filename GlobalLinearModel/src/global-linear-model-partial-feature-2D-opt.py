import datetime
import numpy as np
import random

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


class global_liner_model(object):
    def __init__(self, train_data_file=None, dev_data_file=None, test_data_file=None):
        self.train_data = dataset(train_data_file) if train_data_file != None else None
        self.dev_data = dataset(dev_data_file) if dev_data_file != None else None
        self.test_data = dataset(test_data_file) if test_data_file != None else None
        self.features = {}
        self.weights = []
        self.EOS = 'EOS'
        self.BOS = 'BOS'
        self.v = []
        self.tag2id = {}
        self.id2tag = {}
        self.tags = []

    def create_bigram_feature(self, pre_tag):
        return ['01:' + pre_tag]

    def create_unigram_feature(self, sentence, position):
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

        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])
        return template

    def create_feature_template(self, sentence, position, pre_tag):
        template = []
        template.extend(self.create_bigram_feature(pre_tag))
        template.extend(self.create_unigram_feature(sentence, position))
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
                template = self.create_feature_template(sentence, j, pre_tag)
                for f in template:
                    if f not in self.features:
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
        self.tags = sorted(self.tags)
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.id2tag = {i: t for i, t in enumerate(self.tags)}
        self.weights = np.zeros((len(self.features), len(self.tag2id)))
        self.update_times = np.zeros((len(self.features), len(self.tag2id)))
        self.v = np.zeros((len(self.features), len(self.tag2id)))
        self.bigram_features = [self.create_bigram_feature(prev_tag) for prev_tag in self.tags]
        print("the total number of features is %d" % (len(self.features)))

    def score(self, feature, averaged=False):
        if averaged:
            scores = [self.v[self.features[f]]
                      for f in feature if f in self.features]
        else:
            scores = [self.weights[self.features[f]]
                      for f in feature if f in self.features]
        return np.sum(scores, axis=0)

    def predict(self, sentence, averaged=False):
        states = len(sentence)
        type = len(self.tag2id)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type), dtype='int')

        feature = self.create_bigram_feature(self.BOS)
        feature.extend(self.create_unigram_feature(sentence, 0))
        max_score[0] = self.score(feature, averaged)

        bigram_scores = [
            self.score(f, averaged)
            for f in self.bigram_features
        ]
        for i in range(1, states):
            unigram_feature = self.create_unigram_feature(sentence, i)
            unigram_scores = self.score(unigram_feature, averaged)
            scores = [max_score[i - 1][j] + bigram_scores[j] + unigram_scores
                      for j in range(len(self.tags))]
            paths[i] = np.argmax(scores, axis=0)
            max_score[i] = np.max(scores, axis=0)
        prev = np.argmax(max_score[-1])

        predict = [prev]
        for i in range(len(sentence) - 1, 0, -1):
            prev = paths[i, prev]
            predict.append(prev)
        return [self.tags[i] for i in reversed(predict)]

    def evaluate(self, data, averaged=False):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            predict = self.predict(sentence, averaged)
            for j in range(len(tags)):
                if tags[j] == predict[j]:
                    correct_num += 1

        return correct_num, total_num, correct_num / total_num

    def online_train(self, iteration=20, averaged=False, shuffle=False, exitor=20):
        max_dev_precision = 0
        update_time = 0
        counter = 0
        if averaged:
            print('using V to predict dev data', flush=True)
        for iter in range(iteration):
            print('iterator: %d' % (iter), flush=True)
            if shuffle:
                print('\tshuffle the train data...', flush=True)
                self.train_data.shuffle()
            starttime = datetime.datetime.now()
            bigram_features = [self.create_bigram_feature(pre_tag) for pre_tag in self.tags]
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                predict = self.predict(sentence, False)
                if predict != tags:
                    update_time += 1
                    for j in range(len(sentence)):
                        unigram_feature = self.create_unigram_feature(sentence, j)
                        if j == 0:
                            gold_bigram_feature = self.create_bigram_feature(self.BOS)
                            predict_bigram_feature = self.create_bigram_feature(self.BOS)
                        else:
                            gold_pre_tag = tags[j - 1]
                            predict_pre_tag = predict[j - 1]
                            gold_bigram_feature = bigram_features[self.tag2id[gold_pre_tag]]
                            predict_bigram_feature = bigram_features[self.tag2id[predict_pre_tag]]

                        for f in unigram_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag2id[tags[j]]
                                last_w_value = self.weights[findex][tindex]
                                self.weights[findex][tindex] += 1
                                self.update_v(findex, tindex, update_time, last_w_value)

                                findex = self.features[f]
                                tindex = self.tag2id[predict[j]]
                                last_w_value = self.weights[findex][tindex]
                                self.weights[findex][tindex] -= 1
                                self.update_v(findex, tindex, update_time, last_w_value)

                        for f in gold_bigram_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag2id[tags[j]]
                                last_w_value = self.weights[findex][tindex]
                                self.weights[findex][tindex] += 1
                                self.update_v(findex, tindex, update_time, last_w_value)
                                # self.weights[self.features[f]][self.tag2id[tags[j]]] += 1

                        for f in predict_bigram_feature:
                            if f in self.features:
                                findex = self.features[f]
                                tindex = self.tag2id[predict[j]]
                                last_w_value = self.weights[findex][tindex]
                                self.weights[findex][tindex] -= 1
                                self.update_v(findex, tindex, update_time, last_w_value)

            # 本次迭代完成
            current_update_times = update_time  # 本次更新所在的次数
            for i in range(len(self.v)):
                for j in range(len(self.v[i])):
                    last_w_value = self.weights[i][j]
                    last_update_times = self.update_times[i][j]  # 上一次更新所在的次数
                    if current_update_times != last_update_times:
                        self.update_times[i][j] = current_update_times
                        self.v[i][j] += (current_update_times - last_update_times - 1) * last_w_value + self.weights[i][
                            j]

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data, averaged=False)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision), flush=True)
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data, averaged)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision), flush=True)

            if self.test_data != None:
                test_correct_num, test_num, test_precision = self.evaluate(self.test_data, averaged)
                print('\t' + 'test准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision), flush=True)

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1

            endtime = datetime.datetime.now()
            print("\titeration executing time is " + str((endtime - starttime)) + " s", flush=True)
            if train_correct_num == total_num:
                break
            if counter >= exitor:
                break
        print('iterator = %d , max_dev_precision = %f' % (max_iterator, max_dev_precision), flush=True)

    def update_v(self, findex, tindex, update_time, last_w_value):
        last_update_time = self.update_times[findex][tindex]  # 上一次更新所在的次数
        current_update_time = update_time  # 本次更新所在的次数
        self.update_times[findex][tindex] = update_time
        self.v[findex][tindex] += (current_update_time - last_update_time - 1) * last_w_value + self.weights[findex][
            tindex]


if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    shuffle = config['shuffle']
    exitor = config['exitor']

    starttime = datetime.datetime.now()
    model = global_liner_model(train_data_file, dev_data_file, test_data_file)
    model.create_feature_space()
    print(model.tag2id)
    model.online_train(iterator, averaged, shuffle, exitor)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")
