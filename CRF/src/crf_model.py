from collections import defaultdict
import numpy as np
import random
from scipy.special import logsumexp
from data_loader import DataLoader
from Config import *

class CRFModel:
    def __init__(self, train_set_dir, dev_set_dir, test_set_dir = None):
        self.train_set = DataLoader(train_set_dir) # 训练集对象
        self.dev_set = DataLoader(dev_set_dir)  # 开发集对象
        self.test_set = DataLoader(test_set_dir) if test_set_dir else None # 测试集对象

        self.train_set.display()
        self.dev_set.display()
        if test_set_dir:
            self.test_set.display()
        print()

        self.update_time = 0  # 更新次数（用于更新v权重）
        self.tag_num = self.train_set.tag_num # 词性数目
        self.tags = self.train_set.tags # 词性列表
        self.tag2id = self.train_set.tag2id # 词性-id映射字典
        self.partial_feature_num = 0 # 部分特征数目
        self.partial_feature2id = {} # 部分特征-id映射字典
        self.partial_features = [] # 部分特征列表
        self.weights = np.zeros([self.partial_feature_num, self.tag_num]) # 权重矩阵
        self.bi_gram_features = []  # 二元特征列表
        self.bi_scores = [] # 二元特征得分
        self.create_feature_space() # 开始根据训练集创建特征空间

    @staticmethod
    def create_feature_template(word_list, idx, pre_tag=None, mode='a'):
        """
        创建部分特征模板
        :param mode: 创建特征模板的模式：'a'：既包含uni-gram特征，也包含bi-gram特征;'u'：只包含uni-gram特征，'b'只包含bi-gram特征
        :param pre_tag: 上一个词性
        :param word_list: 句子分词列表
        :param idx: 需要创建模板的词的索引
        :return:
        ft_list：部分特征模板
        """
        ft_list = []  # 特征模板列表
        w = word_list[idx]  # 当前词
        first_c = w[0]  # 当前词第一个字符
        last_c = w[-1]  # 当前词最后一个字符
        prev_w = word_list[idx - 1] if idx > 0 else "**"  # 上一个词
        prev_w_last_c = prev_w[-1]  # 上一个词最后一个字符
        next_w = word_list[idx + 1] if idx + 1 < len(word_list) else "##"  # 下一个词
        next_w_first_c = next_w[0]  # 下一个词第一个字符

        # 添加特征
        if mode == 'b':
            return [('01', pre_tag)]
        elif mode == 'a':
            ft_list.append(('01', pre_tag))  # Bi-gram特征（Global特征，序列标注）
        ft_list.append(('02', w))
        ft_list.append(('03', prev_w))
        ft_list.append(('04', next_w))
        ft_list.append(('05', w, prev_w_last_c))
        ft_list.append(('06', w, next_w_first_c))
        ft_list.append(('07', first_c))
        ft_list.append(('08', last_c))

        for c in w[1:-1]:
            ft_list.append(('09', c))
            ft_list.append(('10', first_c, c))
            ft_list.append(('11', last_c, c))

        if len(w) == 1:
            ft_list.append(('12', w, prev_w_last_c, next_w_first_c))

        for i in range(1, len(w)):
            if w[i - 1] == w[i]:
                ft_list.append(('13', w[i], 'consecutive'))
            if i <= 4:
                ft_list.append(('14', w[:i]))
                ft_list.append(('15', w[-i:]))
        if len(w) <= 4:
            ft_list.append(('14', w))
            ft_list.append(('15', w))

        return ft_list

    def create_feature_space(self):
        """
        创建特征空间（进行特征抽取优化，将定位特征权重的时间复杂度由O(MN)降为O(M+N)）
        :return:
        """
        # 对训练集中每个句子的每个词构建特征模板，并加入特征空间（应避免重复，用set）
        partial_features_set = set()
        for i in range(len(self.train_set.states)):
            word_list = self.train_set.states[i]
            golden_tag_list = self.train_set.golden_tag[i]
            for j in range(len(word_list)):
                for f in self.create_feature_template(word_list, j, golden_tag_list[j-1] if j > 0 else "<BOS>"):
                    partial_features_set.add(f)

        self.partial_features = list(partial_features_set)
        self.bi_gram_features = [[('01', tag)] for tag in self.tags]
        self.bi_scores = np.zeros([self.tag_num, self.tag_num])
        self.partial_feature_num = len(self.partial_features)
        self.partial_feature2id = {pf : idx for idx, pf in enumerate(self.partial_features)}
        self.weights = np.zeros([self.partial_feature_num, self.tag_num])  # w权重矩阵

        print("特征空间维度：{:d}(部分特征数)*{:d}(词性数)".format(self.partial_feature_num,self.tag_num))

    def score(self, ft_list):
        """
        计算某个句子中某词标注为所有词性的得分
        :param ft_list: 该词的特征向量
        :return:
        得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
        """
        f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()]  # 部分特征索引列表
        score_matrix = self.weights[f_id_list]  # 得分矩阵，维度为：部分特征数*词性数
        score_list = np.sum(score_matrix, axis=0)  # 得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
        return score_list

    def propagate(self, word_list, mode = 'f'):
        """
        求解某个句子所有可能的词性序列的得分之和
        :param mode: 选择模式：'f'为前向传播，'b'为后向传播
        :param word_list:
        :return:
        返回一个scores矩阵，维度为：单词数目 * 词性数目，score[i,j]为单词i被标注为第j种词性的所有部分路径的得分总和的对数（后面计算概率需要取对数）
        """
        word_num = len(word_list)  # 当前句子中单词（显状态）个数
        score_matrix = np.zeros([word_num, self.tag_num])  # scores矩阵

        if mode == 'f':
            score_matrix[0] = self.score(self.create_feature_template(word_list, 0, '<BOS>'))
            for i in range(1, word_num):
                uni_scores = self.score(self.create_feature_template(word_list, i, mode='u'))  # 仅考虑一元特征的scores列表
                scores = (self.bi_scores + uni_scores).T + score_matrix[i - 1]  # scores第j行第i列为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前序列的总得分
                score_matrix[i] = logsumexp(scores, axis=1)  # 做一次sum-product（注意：viterbi算法这里是做max-product）
        elif mode == 'b':
            for i in range(word_num - 2, -1 ,-1):
                uni_scores = self.score(self.create_feature_template(word_list, i + 1, mode='u'))  # 仅考虑一元特征的scores列表
                scores = self.bi_scores + uni_scores + score_matrix[i + 1]  # scores第j行第i列为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前序列的总得分
                score_matrix[i] = logsumexp(scores, axis=1)  # 做一次sum-product（注意：viterbi算法这里是做max-product）
        return score_matrix

    def viterbi_predict(self, word_list):
        """
        维特比算法预测整个句子的词性序列
        :param word_list: 句子分词列表
        :return:
        返回该词预测的词性
        """
        word_num = len(word_list)  # 当前句子中单词（显状态）个数
        dp_matrix = np.zeros([word_num, self.tag_num])  # DP状态转移矩阵
        backtrack_matrix = np.zeros([word_num, self.tag_num], dtype='int')  # 回溯矩阵，用于计算路径

        dp_matrix[0] = self.score(self.create_feature_template(word_list, 0, '<BOS>'))
        backtrack_matrix[0] = np.full([self.tag_num], -1) # 第一个词无需回溯路径

        for i in range(1, word_num):
            uni_scores = self.score(self.create_feature_template(word_list, i, mode = 'u')) # 仅考虑一元特征的scores列表
            scores = np.array((self.bi_scores + uni_scores).T + dp_matrix[i-1]) # scores第j行第i列为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前序列的总得分
            backtrack_matrix[i] = np.argmax(scores, axis=1) # 记录状态矩阵中每个节点的值是由上一层的哪个节点转移而来（max-product，即选取score最大者）
            dp_matrix[i] = np.max(scores, axis=1) # 更新当前节点的状态矩阵

        prev = np.argmax(dp_matrix[-1])
        result = [prev]

        # 反向回溯，进行序列标注解码（global seq）
        for i in range(word_num - 1, 0, -1):
            prev = backtrack_matrix[i][prev]
            result.append(prev)

        return list(map(lambda x : self.tags[x],result[::-1]))

    def gradient_descent(self, batch, learning_rate = 0.3, lmbda = 0.01):
        """
        梯度下降法更新模型的权重
        :param lmbda: L2正则化系数
        :param learning_rate: 学习率
        :param batch: 一个batch的样本数据
        :return:
        """
        gradients = defaultdict(float)

        # 在一个batch内对梯度进行累加，一个batch的数据处理完成后一次性对权重进行更新（我感觉这里更像小批次梯度下降而非随机梯度下降，因为SGD每处理完一个样本都会更新一次权重矩阵）
        for sentence in batch:
            word_list, golden_tag_list = zip(*sentence)
            word_list = list(word_list)
            golden_tag_list = list(golden_tag_list)
            word_num = len(word_list)

            pre_tag = '<BOS>'
            for idx, golden_tag in enumerate(golden_tag_list):
                ft_list = self.create_feature_template(word_list, idx, pre_tag)
                f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()]
                for f_id in f_id_list:
                    gradients[f_id, self.tag2id[golden_tag]] -= 1
                pre_tag = golden_tag

            forward_propagate_matrix = self.propagate(word_list, mode = 'f') # 前向传播矩阵
            backward_propagate_matrix = self.propagate(word_list, mode = 'b') # 后向传播矩阵

            log_Z = logsumexp(forward_propagate_matrix[-1]) # 当前句子所有可能的词性标注序列集合的得分总和取对数（这里用后向传播矩阵的首行也是同样的）

            # 更新第一个词对应的特征权重（涉及到BOS标记的转移，需要单独拎出来）
            ft_list = self.create_feature_template(word_list, 0, '<BOS>')
            f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()]
            probs = np.exp(self.score(ft_list) + backward_propagate_matrix[0] - log_Z) # 概率列表：第一个词标注为各词性的概率。概率公式：见讲义4.3。（第一个词的pre_tag均为<BOS>，所以它的probs矩阵只有一行）
            for f_id in f_id_list:
                gradients[f_id] += probs

            # 更新其余词对应的特征权重
            for i in range(1, word_num):
                uni_ft_list = self.create_feature_template(word_list, i, mode='u')
                uni_f_id_list = [self.partial_feature2id[f] for f in uni_ft_list if f in self.partial_feature2id.keys()]
                uni_scores = self.score(uni_ft_list)  # 仅考虑一元特征的scores列表
                scores = np.array(self.bi_scores + uni_scores)  # scores第i行第j列为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前词性的得分（注意：这里与viterbi算法不同，后者是当前序列得分）
                probs = np.exp(forward_propagate_matrix[i-1][:,None] + scores + backward_propagate_matrix[i] - log_Z)  # 概率矩阵：第i行第j列为前一个词的词性为第i种词性、当前词的词性为第j种词性的概率。维度：词性数目 * 词性数目。概率公式：见讲义4.3。
                for bi_ft_list, p in zip(self.bi_gram_features, probs):
                    bi_f_id_list = [self.partial_feature2id[f] for f in bi_ft_list if f in self.partial_feature2id.keys()]
                    for f_id in bi_f_id_list + uni_f_id_list:
                        gradients[f_id] += p

        for key, gradient in gradients.items(): # 梯度下降法更新权重
            self.weights[key] -= learning_rate * (gradient / batch_size + lmbda * self.weights[key])

        self.bi_scores = np.array([self.score(bfv) for bfv in self.bi_gram_features]) # 更新二元特征矩阵

    def evaluate(self, dataset : DataLoader):
        """
        评估模型
        :param dataset: 数据集对象
        :return:
        返回总词数、预测正确的词数、正确率
        """
        total_num = dataset.word_num # 总词数
        correct_num = 0 # 预测正确的词数
        for idx, word_list in enumerate(dataset.states):
            golden_tag_list = dataset.golden_tag[idx] # 标准结果
            predict_tag_list = self.viterbi_predict(word_list) # 预测结果
            correct_num += len([i for i in range(len(golden_tag_list)) if golden_tag_list[i] == predict_tag_list[i]])
        accuracy = correct_num / total_num
        return total_num, correct_num, accuracy

    def mini_batch_train(self, epoch = 100, exitor = 10, random_seed = 0, learning_rate = 0.3, decay_rate = 0.96, lmbda = 0.01, shuffle = True):
        """
        小批量梯度下降进行模型训练
        :param lmbda: L2正则化系数
        :param decay_rate: 学习率衰减速率（用于模拟退火）
        :param learning_rate: 学习率
        :param random_seed: 随机种子
        :param epoch: 迭代总轮数
        :param exitor: 退出轮数
        :param shuffle: 是否打乱数据集
        :return:
        """
        random.seed(random_seed) # 设置随机种子

        global_step = 100000 # 模拟退火最大步数
        max_acc = 0 # 最大准确率（开发集）
        max_acc_epoch = 0 # 最大准确率的轮数（开发集）

        step = 0

        if shuffle: # 打乱训练集（降低过拟合的概率）
            random.shuffle(self.train_set.sentences)

        batches = [self.train_set.sentences[i:i + batch_size] for i in range(0, len(self.train_set.sentences), batch_size)] # 划分batch

        for e in range(1, epoch + 1):
            print("第{:d}轮开始训练...".format(e))

            for batch in batches:
                # 模拟退火，梯度更新次数越多，学习率越小，使模型凸优化时在最优处趋于稳定
                self.gradient_descent(batch, learning_rate * decay_rate ** (step / global_step), lmbda)
                step += 1

            print("本轮训练完成，进行评估...")

            train_total_num, train_correct_num, train_accuracy = self.evaluate(self.train_set)
            print("训练集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(train_total_num, train_correct_num, train_accuracy))

            dev_total_num, dev_correct_num, dev_accuracy = self.evaluate(self.dev_set)
            print("开发集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(dev_total_num, dev_correct_num, dev_accuracy))

            if self.test_set:
                test_total_num, test_correct_num, test_accuracy = self.evaluate(self.test_set)
                print("测试集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(test_total_num, test_correct_num, test_accuracy))

            if dev_accuracy > max_acc:
                max_acc_epoch = e
                max_acc = dev_accuracy
            elif e - max_acc_epoch >= exitor:
                print("经过{:d}轮模型正确率无提升，结束训练！最大正确率为第{:d}轮训练后的{:f}".format(exitor, max_acc_epoch, max_acc))
                break
            print()

CRF = CRFModel(train_data_dir, dev_data_dir)
CRF.mini_batch_train()