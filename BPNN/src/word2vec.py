"""词向量相关类"""
import numpy as np
import pickle
from Config import *
from data_set import DataSet
from data_loader import DataLoader

class Word2Vec:
    def __init__(self, filename, word_embed_dim):
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.unk_id, self.bos_id, self.eos_id = 0, 0, 0
        self.marks = ['<UNK>', '<BOS>', "<EOS>"] # 未登录词、句首、句尾标记
        self.word_embed_dim = word_embed_dim
        self.word_num = 0
        self.word2id,self.id2word,self.embedding_matrix = self.load_word2vec(filename)


    def get_word2vec(self,word):
        """
        根据单词获取词向量
        :param word:
        :return:
        如果单词存在词向量，则返回它的词向量，否则返回-1
        """
        idx = self.word2id[word] if word in self.word2id.keys() else self.word2id['<UNK>']
        return self.embedding_matrix[idx]

    def get_similarity(self,word1,word2):
        """
        计算两个单词的余弦相似度
        :param word1:
        :param word2:
        :return:
        """
        word2vec_1 = self.get_word2vec(word1)
        word2vec_2 = self.get_word2vec(word2)
        return word2vec_1.dot(word2vec_2)/(np.linalg.norm(word2vec_1)*np.linalg.norm(word2vec_2))

    def load_word2vec(self, filename):
        """
        读取词向量
        :param filename: 词向量文件名
        :return:
        word2id:词映射至索引
        id2word:索引映射至词
        word2vec:词向量矩阵
        """
        word2id = {}
        id2word = {}
        word2vec = []
        with open(filename,'rb') as f: # 读取词向量文件
            count = 0
            for line in f:
                tmp = line.decode('utf-8').rstrip('\n').split()
                word2vec.append(list(map(float, tmp[1:])))
                word2id[tmp[0]] = count
                id2word[count] = tmp[0]
                count+=1
        for mark in self.marks:
            if mark not in word2id.keys():
                word2id[mark] = count
                id2word[count] = mark
                word2vec.append((np.random.randn(self.word_embed_dim) / np.sqrt(self.word_embed_dim)).tolist())
        word2vec = np.array(word2vec)
        self.word_num = len(word2id)
        self.eos_id = word2id[self.eos]
        self.bos_id = word2id[self.bos]
        self.unk_id = word2id[self.unk]
        print("初始化词向量矩阵完成！当前词向量矩阵共有{:d}个词，词向量维度为{:d}".format(self.word_num, word_embed_dim))
        return word2id, id2word, word2vec

    def extend_word2vec(self, dataset: DataSet):
        """
        利用数据集扩展词嵌入向量类
        :param dataset: 数据集
        :return:
        """
        unk_words = [word for word in dataset.words if word not in self.word2id.keys()] # 数据集中的未登录词
        len_unk_word = len(unk_words)
        for idx, unk_word in enumerate(unk_words):
            new_idx = idx + self.word_num
            self.word2id[unk_word] = new_idx
            self.id2word[new_idx] = unk_word
        self.embedding_matrix = np.concatenate([self.embedding_matrix, np.random.randn(len_unk_word, self.word_embed_dim) / np.sqrt(self.word_embed_dim)], axis=0) # 数据集中的未登录词加入到词向量矩阵
        self.word_num = len(self.word2id)
        self.eos_id = self.word2id[self.eos]
        self.bos_id = self.word2id[self.bos]
        self.unk_id = self.word2id[self.unk]
        print("通过数据集“{:s}”为词向量矩阵新增{:d}个未登录词，当前词向量矩阵中共有{:d}个词！".format(dataset.name, len_unk_word, self.word_num))

    def save(self, filename):
        """
        保存
        :return:
        """
        with open(filename, "wb") as f:
            pickle.dump(self,f)
