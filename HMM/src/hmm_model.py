# 一阶隐马尔科夫模型

import pickle
import numpy as np
from Config import *
from data_loader import DataLoader


def add_alpha_smooth(matrix):
    """
    加alpha平滑
    :param matrix: 输入矩阵
    :return: 平滑后的矩阵
    """
    for i in range(len(matrix)):
        s = sum(matrix[i])
        for j in range(len(matrix[i])):
            matrix[i][j] = (matrix[i][j] + alpha) / (s + alpha * (len(matrix[0])))
    return np.log(matrix)

class HMM:
    def __init__(self, train_set_dir):
        self.dataset = DataLoader(train_set_dir)  # 训练集对象
        self.word_num = len(self.dataset.words) # 词数
        self.tag_num = len(self.dataset.tags) # 词性数
        self.transition_matrix = np.zeros([self.tag_num, self.tag_num]) # 词性转移矩阵
        self.emit_matrix = np.zeros([self.tag_num-2, self.word_num]) # 词性-词发射矩阵
        self.train() # 开始训练

    def train(self):
        """
        训练模型，统计参数（HMM是生成模型而非判别模型，根据极大似然估计可推导其参数的统计公式）
        :return:
        """
        for sen in self.dataset.sentences:
            length = len(sen) - 1
            for idx, wt in enumerate(sen):
                word, tag = wt
                if idx == 0:
                    self.transition_matrix[self.dataset.tag2id['<BOS>'], self.dataset.tag2id[tag]] += 1 # 句首转移
                elif idx == length:
                    self.transition_matrix[self.dataset.tag2id[tag], self.dataset.tag2id['<EOS>']] += 1 # 句尾转移
                    self.transition_matrix[self.dataset.tag2id[sen[idx - 1][1]], self.dataset.tag2id[tag]] += 1  # 词性转移
                else:
                    self.transition_matrix[self.dataset.tag2id[sen[idx - 1][1]], self.dataset.tag2id[tag]] += 1  # 词性转移

                self.emit_matrix[self.dataset.tag2id[tag], self.dataset.word2id[word]] += 1 # 更新发射矩阵

        # 取对数方便计算，减少过大差异带来的影响，同时采用加alpha平滑，减少稀疏数据带来的影响

        self.transition_matrix = add_alpha_smooth(self.transition_matrix)
        self.emit_matrix = add_alpha_smooth(self.emit_matrix)


    def viterbi_predict(self, sentence):
        """
        维特比算法进行解码预测
        :param sentence: 以列表形式给出分词后的句子
        :return:
        返回标注后的词性序列
        """
        idx_list = list(map(lambda w : self.dataset.word2id.get(w, self.dataset.word2id['<UNK>']), sentence)) # 单词转id映射列表
        word_num = len(sentence)    # 当前句子中单词（显状态）个数
        tag_num = self.tag_num - 2  # 词性（隐状态）个数（不含BOS和EOS）
        dp_matrix = np.zeros([word_num, tag_num]) # DP状态转移矩阵
        backtrack_matrix = np.zeros([word_num, tag_num], dtype='int') # 回溯矩阵，用于计算路径

        # 初始化第一个词的dp矩阵：初始转移概率 + 发射概率
        dp_matrix[0] = self.transition_matrix[self.dataset.tag2id['<BOS>'], :-2] + self.emit_matrix[:, idx_list[0]]

        # 第一个词无需回溯路径
        backtrack_matrix[0] = np.full([tag_num], -1)

        # 前向传播计算各显状态的各隐状态最大得分值
        for i in range(1, word_num):
            score = dp_matrix[i-1].reshape(tag_num, 1) + self.transition_matrix[:-2,:-2]# 计算转移到当前节点的概率 + 上一个节点的状态矩阵值
            backtrack_matrix[i] = np.argmax(score, axis=0) # 记录状态矩阵中每个节点的值是由上一层的哪个节点转移而来（max-product，即选取score最大者）
            dp_matrix[i] = np.max(score + self.emit_matrix[:, idx_list[i]] , axis=0) # 更新当前节点的状态矩阵（需要加上发射概率）

        prev = np.argmax(dp_matrix[-1] + self.transition_matrix[:-2, self.dataset.tag2id['<EOS>']]) # 计算转移到结束标记的概率，取最大值，即为最后一个词的词性最可能的取值
        result = [prev]

        # 反向回溯，进行序列标注解码（global seq）
        for i in range(word_num-1, 0, -1):
            prev = backtrack_matrix[i][prev]
            result.append(prev)

        return list(map(lambda x : self.dataset.tags[x],result[::-1]))

    def evaluate(self, test_set : DataLoader):
        """
        评估测试集
        :param test_set: 测试数据集
        :return:
        """
        correct_num, total_num = 0, 0
        for idx, sen in enumerate(test_set.states):
            res = self.viterbi_predict(sen)
            # print(sen)
            # print(res)
            # print(test_set.golden_tag[idx])
            total_num += len(res)
            correct_num += len([res[j] for j in range(len(res)) if res[j] == test_set.golden_tag[idx][j]])
        print("总句子数：{:d}，总词数：{:d}，预测正确的词数：{:d}，预测正确率：{:f}。".format(len(test_set.states),total_num, correct_num, correct_num / total_num))


    def save(self, save_dir):
        """
        保存模型
        :param save_dir:
        :return:
        """
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)

