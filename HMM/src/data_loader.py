# 数据载入器
import pickle

class DataLoader:
    def __init__(self, dataset_dir):
        self.sentences = [] # 句子列表：(词，词性）
        self.states = [] # 每个句子分词后的列表
        self.golden_tag = [] # 每个句子中所有词对应的词性列表
        self.words = [] # 词列表
        self.tags = [] # 词性列表
        self.tag2id = {} # 词性与id的对应词典
        self.word2id = {} # 词与id的对应词典
        self.load_sentences(dataset_dir)

    def load_sentences(self, dataset_dir):
        """
        读取CoNLL格式数据
        :param dataset_dir: 数据集路径
        :return:
        """
        # 构建句子列表
        with open(dataset_dir, 'r', encoding = 'utf-8') as f:
            self.sentences = []
            word_set = set()
            tag_set = set()
            sen = []
            for line in f:
                li = line.split()
                if li:
                    word, tag = li[1], li[3]
                    word_set.add(word)
                    tag_set.add(tag)
                    sen.append((word, tag))
                else:
                    self.sentences.append(sen)
                    self.states.append([x[0] for x in sen])
                    self.golden_tag.append([x[1] for x in sen])
                    sen = []

        # 构建词列表和词性列表
        self.words = list(sorted(word_set)) + ['<UNK>']
        self.tags = list(sorted(tag_set)) + ['<BOS>', '<EOS>']

        # 构建词典
        self.word2id = {word : idx for idx, word in enumerate(self.words)}
        self.tag2id = {tag : idx for idx, tag in enumerate(self.tags)}

    def save_data(self, save_dir):
        """
        保存处理后的数据集
        :param save_dir:
        :return:
        """
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)




