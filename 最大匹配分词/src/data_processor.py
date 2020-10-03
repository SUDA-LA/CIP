from Config import *

class DataProcessor: #数据预处理类
    def __init__(self):
        self.word_dict, self.sentence = self.read()
        # 由于词典不含重复的单词，所以直接用set比用list更好

    def read(self):
        """
        读取数据
        :return:
        word_dict: 词典
        sentence: 句子
        """
        word_dict = set()
        sentence = []
        sen = []
        with open(data_dir, 'r', encoding='utf-8') as f:
            for line in f:  # 获取词典和句子
                li = line.split()
                if li:
                    self.word_dict.add(li[1] + '\n')
                    sen.append(li[1])
                else:
                    self.sentence.append("".join(sen) + "\n")
                    sen = []
        return word_dict, sentence


    def build_dict(self):
        """
        创建词典
        :return:
        """
        with open(dict_dir, 'w', encoding = 'utf-8') as f:
            f.writelines(self.word_dict) # 输出词典

        print("词典共有{:d}个不重复的词".format(len(self.word_dict)))

    def get_text(self):
        """
        构建毛文本
        :return:
        """
        with open(text_dir, 'w', encoding = 'utf-8') as f:
            f.writelines(self.sentence) # 输出句子

        print("毛文本共有{:d}个句子".format(len(self.sentence)))
