import numpy as np
import random
from data_loader import DataLoader

class GlobalLinearModel:
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
        self.partial_features = [] # 部分特征列表（包含一元和二元特征）
        self.weights = np.zeros([self.partial_feature_num, self.tag_num]) # w权重矩阵
        self.v_weights = np.zeros([self.partial_feature_num, self.tag_num]) # v权重矩阵
        self.update_times = np.zeros([self.partial_feature_num, self.tag_num], dtype='int') # 每个权重更新的时间戳矩阵（用于v权重更新的优化）
        self.bi_gram_features = [] # 二元特征列表
        self.create_feature_space() # 开始根据训练集创建特征空间

    @staticmethod
    def create_feature_template(word_list, idx, pre_tag = None, mode = 'a'):
        """
        创建部分特征模板
        :param mode: 创建特征模板的模式：'a'：既包含uni-gram特征，也包含bi-gram特征;'u'：只包含uni-gram特征，'b'只包含bi-gram特征
        :param pre_tag: 上一个词性
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
        if mode == 'b':
            return [('01', pre_tag)]
        elif mode == 'a':
            ft_list.append(('01', pre_tag)) # Bi-gram特征（Global特征，序列标注）
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
        self.partial_feature_num = len(self.partial_features)
        self.partial_feature2id = {pf : idx for idx, pf in enumerate(self.partial_features)}
        self.weights = np.zeros([self.partial_feature_num, self.tag_num])  # w权重矩阵
        self.v_weights = np.zeros([self.partial_feature_num, self.tag_num])  # v权重矩阵
        self.update_times = np.zeros([self.partial_feature_num, self.tag_num], dtype='int')  # 每个权重更新的时间戳矩阵（用于v权重更新的优化）

        print("特征空间维度：{:d}(部分特征数)*{:d}(词性数)".format(self.partial_feature_num,self.tag_num))

    def score(self, ft_list, avg = False):
        """
        计算某个句子中某词标注为所有词性的得分
        :param ft_list: 该词的特征向量（共16种特征）
        :param avg: 是否使用v权值
        :return:
        得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
        """
        f_id_list = [self.partial_feature2id[f] for f in ft_list if f in self.partial_feature2id.keys()]  # 部分特征索引列表
        score_matrix = self.v_weights[f_id_list] if avg else self.weights[f_id_list]  # 得分矩阵，维度为：部分特征数*词性数
        score_list = np.sum(score_matrix, axis=0)  # 得分列表，行向量，每一个元素都是当前词标注为某个词性的得分
        return score_list

    def viterbi_predict(self, word_list, avg = False):
        """
        维特比算法预测整个句子的词性序列
        :param avg: 是否使用v权值
        :param word_list: 句子分词列表
        :return:
        返回该词预测的词性
        """
        word_num = len(word_list)  # 当前句子中单词（显状态）个数
        dp_matrix = np.zeros([word_num, self.tag_num])  # DP状态转移矩阵
        backtrack_matrix = np.zeros([word_num, self.tag_num], dtype='int')  # 回溯矩阵，用于计算路径
        bi_scores = np.array([self.score(bfv, avg) for bfv in self.bi_gram_features]) # 二元特征得分矩阵


        dp_matrix[0] = self.score(self.create_feature_template(word_list, 0, '<BOS>'), avg)
        backtrack_matrix[0] = np.full([self.tag_num], -1) # 第一个词无需回溯路径

        for i in range(1, word_num):
            uni_scores = self.score(self.create_feature_template(word_list, i, mode = 'u'), avg) # 仅考虑一元特征的scores列表
            scores = np.array((dp_matrix[i-1] + bi_scores.T).T + uni_scores) # scores第i行第j列为上一时间步的第i种词性转移到当前时间步的第j种词性后，当前序列的总得分
            backtrack_matrix[i] = np.argmax(scores, axis=0) # 记录状态矩阵中每个节点的值是由上一层的哪个节点转移而来（max-product，即选取score最大者）
            dp_matrix[i] = np.max(scores, axis=0) # 更新当前节点的状态矩阵

        prev = np.argmax(dp_matrix[-1])
        result = [prev]

        # 反向回溯，进行序列标注解码（global seq）
        for i in range(word_num - 1, 0, -1):
            prev = backtrack_matrix[i][prev]
            result.append(prev)

        return list(map(lambda x : self.tags[x],result[::-1]))


    def update(self, sen, avg):
        """
        实时更新模型的权重
        :param avg: 是否使用v权重
        :param sen: 句子列表
        :return:
        """
        word_list, golden_tag_list = [], []
        for t in sen:
            word_list.append(t[0])
            golden_tag_list.append(t[1])
        predict_tag_list = self.viterbi_predict(word_list, False) # 预测整个句子的词性列表
        if predict_tag_list != golden_tag_list: # 如果至少一个词的词性预测失败，则更新权值
            self.update_time += 1
            for i in range(len(sen)):
                if predict_tag_list[i] != golden_tag_list[i]: # 对于每个预测错词性的词，更新对应的权重
                    # 更新uni-gram特征权重
                    predict_tag_id = self.tag2id[predict_tag_list[i]]
                    golden_tag_id = self.tag2id[golden_tag_list[i]]
                    unigram_ft_list = self.create_feature_template(word_list, i, mode = 'u') # 一元部分特征列表
                    unigram_f_id_list = [self.partial_feature2id[f] for f in unigram_ft_list if f in self.partial_feature2id.keys()] # 一元部分特征索引列表
                    for f_id in unigram_f_id_list:  # 对正确词性的特征权重全部加1（激励），对错误词性的特征权值全部减1（惩罚）
                        if avg:  # 使用v权重（我感觉v权重有些类似于SGD中的momentum梯度下降，和指数加权平均数的作用一样，降低每一次学习的影响）
                            last_w = self.weights[f_id, predict_tag_id]  # 暂存w权重
                            self.weights[f_id, predict_tag_id] -= 1
                            self.v_weights[f_id, predict_tag_id] += (self.update_time - self.update_times[f_id, predict_tag_id] - 1) * last_w + self.weights[f_id, predict_tag_id]  # 每一次都对v权重累加w权重的值会很低效，所以我们改为在权重有变化时一次性更新v权重（使用乘法）
                            self.update_times[f_id, predict_tag_id] = self.update_time  # 更新时间戳

                            last_w = self.weights[f_id, golden_tag_id]
                            self.weights[f_id, golden_tag_id] += 1
                            self.v_weights[f_id, golden_tag_id] += (self.update_time - self.update_times[f_id, golden_tag_id] - 1) * last_w + self.weights[f_id, golden_tag_id]
                            self.update_times[f_id, golden_tag_id] = self.update_time
                        else:  # 不使用v权重
                            self.weights[f_id, predict_tag_id] -= 1
                            self.weights[f_id, golden_tag_id] += 1

                    # 更新bi-gram特征权重（这里只需要更新：（1）预测序列中前一个词性到当前错误词性的转移特征减1（2）正确序列中前一个词性到当前错误词性的转移特征加1。其余不需要改变)
                    predict_pre_tag_feature = ('01', predict_tag_list[i - 1] if i > 0 else '<BOS>')
                    golden_pre_tag_feature = ('01', golden_tag_list[i - 1] if i > 0 else '<BOS>')
                    if avg:
                        if predict_pre_tag_feature in self.partial_feature2id.keys():
                            predict_pre_tag_feature_id = self.partial_feature2id[predict_pre_tag_feature]
                            last_w = self.weights[predict_pre_tag_feature_id, predict_tag_id]  # 暂存w权重
                            self.weights[predict_pre_tag_feature_id, predict_tag_id] -= 1
                            self.v_weights[predict_pre_tag_feature_id, predict_tag_id] += (self.update_time - self.update_times[predict_pre_tag_feature_id, predict_tag_id] - 1) * last_w + self.weights[predict_pre_tag_feature_id, predict_tag_id]  # 每一次都对v权重累加w权重的值会很低效，所以我们改为在权重有变化时一次性更新v权重（使用乘法）
                            self.update_times[predict_pre_tag_feature_id, predict_tag_id] = self.update_time  # 更新时间戳
                        if golden_pre_tag_feature in self.partial_feature2id.keys():
                            golden_pre_tag_feature_id = self.partial_feature2id[golden_pre_tag_feature]
                            last_w = self.weights[golden_pre_tag_feature_id, golden_tag_id]  # 暂存w权重
                            self.weights[golden_pre_tag_feature_id, golden_tag_id] += 1
                            self.v_weights[golden_pre_tag_feature_id, golden_tag_id] += (self.update_time - self.update_times[golden_pre_tag_feature_id, golden_tag_id] - 1) * last_w + self.weights[golden_pre_tag_feature_id, golden_tag_id]  # 每一次都对v权重累加w权重的值会很低效，所以我们改为在权重有变化时一次性更新v权重（使用乘法）
                            self.update_times[golden_pre_tag_feature_id, golden_tag_id] = self.update_time  # 更新时间戳
                    else:
                        if predict_pre_tag_feature in self.partial_feature2id.keys():
                            predict_pre_tag_feature_id = self.partial_feature2id[predict_pre_tag_feature]
                            self.weights[predict_pre_tag_feature_id, predict_tag_id] -= 1
                        if golden_pre_tag_feature in self.partial_feature2id.keys():
                            golden_pre_tag_feature_id = self.partial_feature2id[golden_pre_tag_feature]
                            self.weights[golden_pre_tag_feature_id, golden_tag_id] += 1


    def evaluate(self, dataset : DataLoader, avg):
        """
        评估模型
        :param dataset: 数据集对象
        :param avg: 是否使用v权重
        :return:
        返回总词数、预测正确的词数、正确率
        """
        total_num = dataset.word_num # 总词数
        correct_num = 0 # 预测正确的词数
        for idx, word_list in enumerate(dataset.states):
            golden_tag_list = dataset.golden_tag[idx] # 标准结果
            predict_tag_list = self.viterbi_predict(word_list, avg) # 预测结果
            correct_num += len([i for i in range(len(golden_tag_list)) if golden_tag_list[i] == predict_tag_list[i]])
        accuracy = correct_num / total_num
        return total_num, correct_num, accuracy

    def online_train(self, epoch, exitor, avg = False, shuffle = True):
        """
        在线学习（推荐系统中常用的一种学习方法，每次选取一个样本更新模型权重，可以实时更新模型）
        :param epoch: 迭代总轮数
        :param exitor: 退出轮数
        :param avg: 是否使用v权重
        :param shuffle: 是否打乱数据集
        :return:
        """
        max_acc = 0 # 最大准确率（开发集）
        max_acc_epoch = 0 # 最大准确率的轮数（开发集）

        if shuffle: # 打乱训练集（降低过拟合的概率）
            random.shuffle(self.train_set.sentences)

        self.update_time = 0  # 更新次数（用于更新v权重）
        for e in range(1, epoch + 1):
            print("第{:d}轮开始训练...".format(e))

            for sen in self.train_set.sentences:
                self.update(sen, avg)

            # 每轮结束要对v权重进行一次更新（这个好像在讲义中没有提到）
            self.v_weights += [(self.update_time - r) * w for r, w in zip(self.update_times, self.weights)]
            self.update_times = np.full([self.partial_feature_num, self.tag_num], self.update_time)

            print("本轮训练完成，进行评估...")

            train_total_num, train_correct_num, train_accuracy = self.evaluate(self.train_set, avg)
            print("训练集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(train_total_num, train_correct_num, train_accuracy))

            dev_total_num, dev_correct_num, dev_accuracy = self.evaluate(self.dev_set, avg)
            print("开发集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(dev_total_num, dev_correct_num, dev_accuracy))

            if self.test_set:
                test_total_num, test_correct_num, test_accuracy = self.evaluate(self.test_set, avg)
                print("测试集共有{:d}个词，预测正确了{:d}个词的词性，正确率为:{:f}".format(test_total_num, test_correct_num, test_accuracy))

            if dev_accuracy > max_acc:
                max_acc_epoch = e
                max_acc = dev_accuracy
            elif e - max_acc_epoch >= exitor:
                print("经过{:d}轮模型正确率无提升，结束训练！最大正确率为第{:d}轮训练后的{:f}".format(exitor, max_acc_epoch, max_acc))
                break
            print()