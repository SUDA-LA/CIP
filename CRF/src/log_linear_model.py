from collections import defaultdict
import numpy as np
import random
from scipy.special import logsumexp
from data_loader import DataLoader
from Config import *

class LogLinearModel:
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
        self.create_feature_space() # 开始根据训练集创建特征空间

    @staticmethod
    def create_feature_template(word_list, idx):
        """
        创建部分特征模板
        :param word_list: 句子分词列表
        :param idx: 需要创建模板的词的索引
        :return:
        ft_list：部分特征模板
        """
        ft_list = [] # 特征模板列表
        w = word_list[idx] # 当前词
        first_c = w[0] # 当前词第一个字符
        last_c = w[-1] # 当前词最后一个字符
        prev_w = word_list[idx - 1] if idx > 0 else "**" # 上一个词
        prev_w_last_c = prev_w[-1] # 上一个词最后一个字符
        next_w = word_list[idx + 1] if idx + 1 < len(word_list) else "##" # 下一个词
        next_w_first_c = next_w[0] # 下一个词第一个字符

        # 添加特征
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
        for word_list in self.train_set.states:
            for i in range(len(word_list)):
                for f in self.create_feature_template(word_list, i):
                    partial_features_set.add(f)

        self.partial_features = list(partial_features_set)
        self.partial_feature_num = len(self.partial_features)
        self.partial_feature2id = {pf : idx for idx, pf in enumerate(self.partial_features)}
        self.weights = np.zeros([self.partial_feature_num, self.tag_num])  # w权重矩阵

        print("特征空间维度：{:d}(部分特征数)*{:d}(词性数)".format(self.partial_feature_num,self.tag_num))

    def predict(self, word_list, idx):
        """
        计算得分并预测给定句子第idx个词的词性
        :param word_list: 句子分词列表
        :param idx: 单词id
        :return:
        tag: 返回该词预测的词性
        """
        ft_list = self.create_feature_template(word_list, idx) # 创建部分特征模板列表
        f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()] # 部分特征索引列表
        score_matrix = self.weights[f_id_list] # 得分矩阵，维度为：在特征空间中的部分特征数*词性数
        score_list = np.sum(score_matrix, axis=0) # 得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
        return self.tags[np.argmax(score_list)]

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
        for word, ft_list, golden_tag in batch:
            f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()]  # 部分特征索引列表
            score_matrix = self.weights[f_id_list]  # 得分矩阵，维度为：在特征空间中的部分特征数*词性数
            score_list = np.sum(score_matrix, axis=0)  # 得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
            probs = np.exp(score_list - logsumexp(score_list)) # 得分转概率（softmax归一化函数两侧同取对数可推导）

            # 梯度计算：gradient = dLoss/dw , 损失函数对各权重求偏导数可得。由于词性标注的特征数目太多，且均为one-hot编码，所以有大量梯度为0，直接对指定梯度进行更新更快。
            for f_id in f_id_list:
                gradients[f_id] += probs
                gradients[f_id, self.tag2id[golden_tag]] -= 1

        for key, gradient in gradients.items(): # 梯度下降法更新权重
            self.weights[key] -= learning_rate * (gradient / batch_size + lmbda * self.weights[key])

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
            for i in range(len(word_list)):
                if self.predict(word_list, i) == golden_tag_list[i]:
                    correct_num += 1
        accuracy = correct_num / total_num
        return total_num, correct_num, accuracy

    def mini_batch_train(self, epoch = 100, exitor = 10, random_seed = 0, learning_rate = 0.3, decay_rate = 0.96, lmbda = 0.01, shuffle = True):
        """
        在线学习（推荐系统中常用的一种学习方法，每次选取一个样本更新模型权重，可以实时更新模型）
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

        max_acc = 0 # 最大准确率（开发集）
        max_acc_epoch = 0 # 最大准确率的轮数（开发集）

        step = 0
        train_data = [] # 训练集
        for sen in self.train_set.sentences:
            word_list, golden_tag_list = zip(*sen)
            word_list, golden_tag_list = list(word_list), list(golden_tag_list)
            for idx, word in enumerate(word_list):
                ft_list = self.create_feature_template(word_list, idx)
                train_data.append((word, ft_list, golden_tag_list[idx]))

        if shuffle: # 打乱训练集（降低过拟合的概率）
            random.shuffle(train_data)

        batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)] # 划分batch

        for e in range(1, epoch + 1):
            print("第{:d}轮开始训练...".format(e))

            for batch in batches:
                # 模拟退火，梯度更新次数越多，学习率越小，使模型凸优化时在最优处趋于稳定
                self.gradient_descent(batch, learning_rate * decay_rate ** (step / 100000), lmbda)
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