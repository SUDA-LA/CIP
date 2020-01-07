import numpy as np
import datetime
from itertools import chain
from config import config

train_data_file = config['train_data_file']
test_data_file = config['test_data_file']
predict_file = config['predict_file']
alpha = config['alpha']


def read_data(filename):
    f = open(filename, encoding='utf-8')
    data = []
    sentence = []
    while (True):
        line = f.readline()
        if not line:
            break
        if line != '\n':
            word = line.split()[1]
            tag = line.split()[3]
            sentence.append((word, tag))
        else:
            data.append(sentence)
            sentence = []
    f.close()
    return data


class Binary_HMM(object):
    # 属性1：train_data存放所有的训练句子，[[(戴相龙,NR),(,),(,)....],[],[]....]
    # 属性2：tag_dict存放训练集中所有的tag，及其编号,考虑了起始和终止词性
    # 属性3：word_dict存放训练集中所有的word，及其编号,加入了未知词
    # 属性4：transition_matrix转移概率矩阵,第(i,j)个元素表示词性j在词性i后面的概率,最后一行是start，最后一列是stop
    # 属性5: launch_matrix发射概率矩阵,第(i,j)个元素表示词性i发射到词j的概率,最后一列是未知词
    def __init__(self, train_data):
        self.train_data = train_data

        sen_seq, tag_seq = zip(*(chain(*train_data)))
        word_list = list(sorted(set(sen_seq)))
        tag_list = list(sorted(set(tag_seq)))
        word_list.append('<UNK>')
        tag_list.append('<BOS>')
        tag_list.append('<EOS>')

        self.tag_dict = {tag: id for id, tag in enumerate(tag_list)}
        self.word_dict = {word: id for id, word in enumerate(word_list)}

        self.transition_matrix = np.zeros(
            [len(self.tag_dict) - 1, len(self.tag_dict) - 1])  # 第(i,j)个元素表示词性j在词性i后面的概率,最后一行是start，最后一列是stop
        self.launch_matrix = np.zeros(
            [len(self.tag_dict) - 2, len(self.word_dict)])  # 第(i,j)个元素表示词性i发射到词j的概率

    def launch_params(self, alpha):
        for sentence in self.train_data:
            for word, tag in sentence:
                self.launch_matrix[self.tag_dict[tag]
                                   ][self.word_dict[word]] += 1
        for i in range(len(self.launch_matrix)):
            s = sum(self.launch_matrix[i])
            for j in range(len(self.launch_matrix[i])):
                self.launch_matrix[i][j] = (
                    self.launch_matrix[i][j] + alpha) / (s + alpha * (len(self.word_dict)))

    def transition_params(self, alpha):
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i]) + 1):
                if j == 0:
                    self.transition_matrix[-1][self.tag_dict[train_data[i][j][1]]] += 1
                elif j == len(self.train_data[i]):
                    self.transition_matrix[self.tag_dict[train_data[i]
                                                         [j - 1][1]]][-1] += 1
                else:
                    self.transition_matrix[self.tag_dict[train_data[i][j - 1][1]]][
                        self.tag_dict[train_data[i][j][1]]] += 1

        for i in range(len(self.transition_matrix)):
            s = sum(self.transition_matrix[i])
            for j in range(len(self.transition_matrix[i])):
                self.transition_matrix[i][j] = (self.transition_matrix[i][j] + alpha) / (
                    s + alpha * (len(self.tag_dict) - 1))
                # self.transition_matrix[i][j]/=s

    def write_matrix(self, matrix, path):
        f = open(path, 'w', encoding='utf-8')
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                f.write('%-10.8f ' % (matrix[i][j]))
            f.write('\n')
        f.close()

    def viterbi(self, word_list):
        word_index = list(map(lambda x: self.word_dict.get(
            x, self.word_dict['<UNK>']), word_list))

        states = len(word_list)
        type = len(self.tag_dict) - 2
        max_p = np.zeros((states, type))
        path = np.zeros((states, type))

        launch_matrix = np.log(self.launch_matrix)
        transition_matrix = np.log(self.transition_matrix)

        # 初始化起始状态
        max_p[0] = transition_matrix[-1, :-1]+launch_matrix[:, word_index[0]]
        path[0] = np.full((type), -1)

        # 动态规划
        for i in range(1, states):
            score = transition_matrix[:-1, :-1]+launch_matrix[:,
                    word_index[i]]+max_p[i-1].reshape(type, 1)
            max_p[i] = np.max(score, axis=0)
            path[i] = np.argmax(score, axis=0)

        max_p[-1] += transition_matrix[:-1, -1]

        prev = np.argmax(max_p[-1])
        predict = [prev]

        for i in range(states-1, 0, -1):
            prev = int(path[i][prev])
            predict.insert(0, prev)
        return predict

    def evaluate(self, test_data):
        total_words = 0
        correct_words = 0
        sentence_num = 0
        print('正在评估测试集...')
        f = open(predict_file, 'w', encoding='utf-8')
        for sentence in test_data:
            sentence_num += 1
            # print('正在预测第%d个句子' % (sentence_num))
            word_list = []
            tag_list = []
            for word, tag in sentence:
                word_list.append(word)
                tag_list.append(tag)
            predict = self.viterbi(word_list)
            total_words += len(sentence)
            for i in range(len(predict)):
                f.write(word_list[i] + '	_	' +
                        list(self.tag_dict.keys())[predict[i]] + '\n')
                if predict[i] == self.tag_dict[tag_list[i]]:
                    correct_words += 1
            f.write('\n')
        f.close()
        print('共%d个句子' % (sentence_num))
        print('共%d个单词，预测正确%d个单词' % (total_words, correct_words))
        print('准确率：%f' % (correct_words / total_words))


if __name__ == '__main__':
    train_data = read_data(train_data_file)
    HMM = Binary_HMM(train_data)
    HMM.launch_params(alpha)
    HMM.transition_params(alpha)
    test_data = read_data(test_data_file)
    starttime = datetime.datetime.now()
    HMM.evaluate(test_data)
    endtime = datetime.datetime.now()
    print('共耗时' + str(endtime - starttime))
