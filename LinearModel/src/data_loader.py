# 数据载入器
import pickle

class DataLoader:
    def __init__(self, dataset_dir):
        self.name = dataset_dir.split('/')[-1]
        self.sentences = [] # 句子列表：(词，词性）
        self.states = [] # 每个句子分词后的列表
        self.tags = [] # 词性列表
        self.tag2id = {}  # 词性与id的对应词典
        self.golden_tag = [] # 每个句子中所有词对应的词性列表
        self.word_num = 0 # 总词数
        self.sentence_num = 0 # 总句子数
        self.tag_num = 0 # 总词性数
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
            self.word_num = 0
            self.sentence_num = 0
            tag_set = set()
            sen = []
            for line in f:
                li = line.split()
                if li:
                    word, tag = li[1], li[3]
                    sen.append((word, tag))
                    tag_set.add(tag)
                    self.word_num += 1
                else:
                    self.sentences.append(sen)
                    self.states.append([x[0] for x in sen])
                    self.golden_tag.append([x[1] for x in sen])
                    sen = []
                    self.sentence_num += 1
        self.tags = list(sorted(tag_set))
        self.tag_num = len(self.tags)
        self.tag2id = {tag: idx for idx, tag in enumerate(self.tags)}

    def display(self):
        """
        展示数据集内容
        :return:
        """
        print("数据集：{:s}，句子数：{:d}，词数：{:d}，词性数目：{:d}".format(self.name, self.sentence_num, self.word_num, self.tag_num))

    def save_data(self, save_dir):
        """
        保存处理后的数据集
        :param save_dir:
        :return:
        """
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)




