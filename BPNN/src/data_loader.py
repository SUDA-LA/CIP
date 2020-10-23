# 数据载入器类
import numpy as np
import random

class DataLoader:
    def __init__(self, dataset, tag2id, word2vec, shuffle = True, batch = False, window = 5, batch_size = 32, tag_num = 31):
        self.dataset = dataset # 数据集对象
        self.word2vec = word2vec # word2vec数据对象
        self.window = window # 上下文窗口大小
        self.tag_num = tag_num # 词性数目
        self.tag2id = tag2id
        self.data = [] # 数据
        self.load() # 处理数据
        if shuffle:
            self.shuffle() # 打乱数据
        if batch:
            self.batches = self.batch(batch_size) # 划分batch

    def shuffle(self):
        """
        随机打乱数据
        :return:
        """
        print("正在打乱：{:s}...".format(self.dataset.name))
        random.shuffle(self.data)

    def batch(self, batch_size):
        """
        划分batch
        :param batch_size: batch内的样本数目
        :return:
        """
        print("正在划分batch：{:s}，batch大小：{:d}...".format(self.dataset.name, batch_size))
        return [self.data[k:k + batch_size] for k in range(0, len(self.data), batch_size)]

    def load(self):
        """
        根据数据集对象和词嵌入矩阵对象，生成数据(X,Y)，
        X是每个词样本最终传入神经网络的表示（word2vec采样window个上下位单词，获取它们的id）
        Y是每个词样本最终的正确标注结果
        :return:
        """
        half = self.window // 2
        for sen in self.dataset.sentences:
            word_list, golden_tag_list = zip(*sen)
            word_list, golden_tag_list = list(word_list), list(golden_tag_list)
            word_id_list = [self.word2vec.bos_id] * half + [self.word2vec.word2id.get(word, self.word2vec.unk_id) for word in word_list] + [self.word2vec.eos_id] * half # 这里需要在两端进行PAD操作，以便窗口能被完整地取到
            golden_tag_id_list = [self.tag2id[tag] for tag in golden_tag_list]
            for idx, tag_id in enumerate(golden_tag_id_list):
                sample_x = word_id_list[idx: idx + self.window]
                sample_y = self.one_hot(tag_id)
                self.data.append((sample_x, sample_y))
        print("{:s}数据集载入完成！样本规模：{:d}".format(self.dataset.name, len(self.data)))

    def one_hot(self, tag_id):
        """
        根据词性id生产独热编码
        :param tag_id: 词性id
        :return:
        """
        vector = np.zeros([self.tag_num, 1])
        vector[tag_id] = 1.0
        return vector
