from collections import defaultdict
import numpy as np
import random
from scipy.special import logsumexp
from data_set import DataSet
from utils import *
from Config import *
from word2vec import Word2Vec
from data_loader import DataLoader

class BPNNModel:
    def __init__(self, train_set_dir, dev_set_dir, test_set_dir, layer_sizes, word2vec_dir, word_embed_dim):
        self.train_set = DataSet(train_set_dir) # 训练集对象
        self.dev_set = DataSet(dev_set_dir)  # 开发集对象
        self.test_set = DataSet(test_set_dir) if test_set_dir else None # 测试集对象


        self.word2vec = Word2Vec(word2vec_dir, word_embed_dim)
        self.word2vec.extend_word2vec(self.train_set)

        self.train_set.display()
        self.dev_set.display()
        if test_set_dir:
            self.test_set.display()
        print()

        self.tag_num = self.train_set.tag_num # 词性数目
        self.tags = self.train_set.tags # 词性列表
        self.tag2id = self.train_set.tag2id # 词性-id映射字典
        self.layer_sizes = layer_sizes # 神经网络每一层的神经元数目（如果样本输入采用独热编码，将会非常稀疏，浪费空间，所以将embedding层单独拎出来更新）
        self.layer_sizes.append(self.tag_num) # 添加输出层维度
        self.layer_num = len(self.layer_sizes)
        self.embed = self.word2vec.embedding_matrix # 词嵌入层
        self.embed_dim = self.word2vec.word_embed_dim # 词嵌入维度

        assert self.tag_num == self.layer_sizes[-1], "数据集词性数目与输出层神经元个数不匹配"

        self.weights = [np.random.randn(l2, l1) / np.sqrt(l1) for l1, l2 in zip(self.layer_sizes[:-1], self.layer_sizes[1:])] # 神经网络权重矩阵的列表
        self.biases = [np.random.randn(l, 1) for l in self.layer_sizes[1:]] # 偏差项列表（神经网络相邻两层可以看作是多个逻辑回归模型的组合）

    def forward_prop(self, x, activation):
        """
        前向传播过程
        :param activation: 激活函数
        :param x: 样本输入
        :return:
        z_list：存储每一层神经元input的列表
        a_list：存储每一层神经元output的列表
        """
        a = np.reshape(self.embed[x], (-1, 1)) # 样本输入通过embedding层，得到维度为：(window * word_embed_dim, 1)的激活值
        # 理论上这一步也可以通过矩阵乘法实现，但会导致输入层的维度过大，且输入非常稀疏
        z_list, a_list = [], [a] # 存储每层神经元的input-Z和output-A（激活后）的列表
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b # 全连接层前向传播
            z_list.append(z) # 缓存本层Z
            if activation == 'sigmoid': # sigmoid激活
                a = sigmoid(z)
            elif activation == 'ReLU': # ReLU激活
                a = relu(z)
            a_list.append(a) # 缓存本层A
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        z_list.append(z)
        a = softmax(z) # 输出层利用softmax激活函数完成概率分布的转化
        a_list.append(a)
        return z_list, a_list

    def backward_prop(self, z_list, a_list, y, activation):
        """
        反向传播过程（计算梯度）
        :param z_list: 存储每一层神经元input的列表
        :param a_list: 存储每一层神经元output的列表
        :param activation: 激活函数
        :param y: 标签数据
        :return:
        d_b: 各偏差项的梯度
        d_w: 各权重的梯度
        d_x: 词嵌入层梯度（仅包含样本中的词）
        """
        d_b = [np.zeros(b.shape) for b in self.biases] # 各偏差项的梯度
        d_w = [np.zeros(w.shape) for w in self.weights] # 各权重的梯度

        d_z = a_list[-1] - y # 交叉熵损失函数对softmax层的输出a求导可得
        d_b[-1] = d_z
        d_w[-1] = np.dot(d_z, a_list[-2].T) # 链式法则，dLoss/dw_(-1) = dLoss/dz_(-1) * dz_(-1)/dw_(-1)，后者等于a_(-2)

        for l in range(2, self.layer_num): # 链式法则进行动态规划反向传播
            z = z_list[-l]
            d_a = np.dot(self.weights[-l + 1].T, d_z)
            if activation == 'sigmoid': # sigmoid激活
                d_z =  d_a * sigmoid_backward(z)
            elif activation == 'ReLU': # ReLU激活
                d_z =  d_a * relu_backward(z)
            d_b[-l] = d_z
            d_w[-l] = np.dot(d_z, a_list[-l - 1].T)
        d_x = np.dot(self.weights[0].T, d_z)
        return d_b, d_w, d_x

    def gradient_descent(self, batch, batch_size, learning_rate, lmbda, n, activation, embedding_freeze = False):
        """
        梯度下降更新网络权重
        :param activation: 激活函数
        :param embedding_freeze: 冻结词向量层
        :param batch: 样本数据
        :param batch_size: batch规模
        :param learning_rate: 学习率
        :param lmbda: L2正则化系数
        :param n: 样本数
        :return:
        """
        d_b = [np.zeros(b.shape) for b in self.biases] # 偏差项梯度矩阵
        d_w = [np.zeros(w.shape) for w in self.weights] # 神经网络权重梯度矩阵
        d_x = defaultdict(float) # 由于embedding层权重规模极大，且梯度非常稀疏，宜采用dict方式更新

        for x, y in batch:
            z_list, a_list = self.forward_prop(x, activation) # 前向传播，计算每层神经元的输入/输出
            d_b_new, d_w_new, d_x_new = self.backward_prop(z_list, a_list, y, activation) # 反向传播，计算所有权重的梯度
            d_b = [d_b[i] + d_b_new[i] for i in range(len(d_b))] # 同一个batch内的b梯度累加
            d_w = [d_w[i] + d_w_new[i] for i in range(len(d_w))] # 同一个batch内的w梯度累加
            for word_id, g in zip(x, np.reshape(d_x_new, (-1, self.embed_dim))): # 将词嵌入层梯度打包成(词id,词嵌入梯度)的形式，利用dict机制更新对应词的词向量权重
                d_x[word_id] += g
        self.weights = [w - (w * learning_rate * lmbda / n) - (learning_rate * grad / batch_size) for w, grad in zip(self.weights, d_w)] # 梯度下降法更新模型参数
        self.biases = [b - (learning_rate * grad / batch_size)for b, grad in zip(self.biases, d_b)] # 梯度下降法更新模型参数
        if not embedding_freeze:
            for word_id, grad in d_x.items(): # 更新embedding层
                self.embed[word_id] -= (learning_rate * grad / batch_size)
        return self.compute_loss(batch, lmbda, activation)

    def compute_loss(self, data, lmbda, activation):
        """
        计算交叉熵损失
        :param activation: 激活函数
        :param data: 数据
        :param lmbda: 正则化系数
        :return:
        """
        loss = 0.0
        for x, y in data:
            _, a_list = self.forward_prop(x, activation)
            a = a_list[-1] # 获取softmax层的输出
            loss -= np.log(a[np.argmax(y)]) # 根据Cross-Entropy Loss的公式计算Loss
        loss += lmbda * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        loss /= 2 * len(data)
        return loss

    def evaluate(self, data, activation):
        """
        评估模型
        :param activation: 激活函数
        :param data: 数据集
        :return:
        返回总词数、预测正确的词数、正确率
        """
        total_num = len(data) # 总token数
        predict_result = [] # 预测结果
        for x, y in data:
            _, a_list = self.forward_prop(x, activation)
            a = a_list[-1]  # 获取softmax层的输出
            predict_result.append(y[np.argmax(a)])
        correct_num = int(sum(predict_result)[0]) # 预测正确的token数
        accuracy = correct_num / total_num # 准确率
        return total_num, correct_num, accuracy

    def mini_batch_train(self, epoch = 100, exitor = 10, random_seed = 0, batch_size = 32, learning_rate = 0.3, decay_rate = 0.96, lmbda = 0.01, window = 5, shuffle = True, activation = 'sigmoid', embedding_freeze = False):
        """
        小批量梯度下降进行模型训练
        :param window: 上下位窗口大小
        :param embedding_freeze:
        :param activation:
        :param batch_size: batch规模
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

        train_loader = DataLoader(self.train_set, self.tag2id, self.word2vec, shuffle, True, window, batch_size, self.tag_num)
        dev_loader = DataLoader(self.dev_set, self.tag2id, self.word2vec, False, False, window, batch_size, self.tag_num)
        test_loader = None
        if self.test_set:
            test_loader = DataLoader(self.test_set, self.tag2id, self.word2vec, False, False, window, batch_size, self.tag_num)

        step = 0
        n = len(train_loader.batches)

        for e in range(1, epoch + 1):
            print("第{:d}轮开始训练...".format(e))

            for batch in train_loader.batches:
                # 模拟退火，梯度更新次数越多，学习率越小，使模型凸优化时在最优处趋于稳定
                self.gradient_descent(batch, batch_size,learning_rate * decay_rate ** (step / global_step), lmbda, n, activation, embedding_freeze)
                step += 1

            print("本轮训练完成，进行评估...")

            train_total_num, train_correct_num, train_accuracy = self.evaluate(train_loader.data, activation)
            print("训练集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(train_total_num, train_correct_num, train_accuracy))

            dev_total_num, dev_correct_num, dev_accuracy = self.evaluate(dev_loader.data, activation)
            print("开发集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(dev_total_num, dev_correct_num, dev_accuracy))

            if self.test_set:
                test_total_num, test_correct_num, test_accuracy = self.evaluate(test_loader.data, activation)
                print("测试集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(test_total_num, test_correct_num, test_accuracy))

            if dev_accuracy > max_acc:
                max_acc_epoch = e
                max_acc = dev_accuracy
            elif e - max_acc_epoch >= exitor:
                print("经过{:d}轮模型正确率无提升，结束训练！最大正确率为第{:d}轮训练后的{:f}".format(exitor, max_acc_epoch, max_acc))
                break
            print()