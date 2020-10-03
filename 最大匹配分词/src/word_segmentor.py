from Config import *

class WordSegmentor: # 分词器类
    def __init__(self):
        self.word_dict = set()
        self.sentence = []
        self.len = 0
        self.out_words = []
        self.correct_words = []
        self.load_data()

    def load_data(self):
        """
        读取数据
        :return:
        """
        self.word_dict, self.sentence, self.len = self.get_data()
        with open(data_dir, 'r', encoding = 'utf-8') as f:
            for line in f:
                li = line.split()
                if li:
                    self.correct_words.append(li[1])

    def get_data(self):
        """
        获取词典和毛文本，以及最大词长
        :return:
        word_dict: 词典
        sentence: 句子
        max_len: 最大词长
        """
        word_dict = set()
        sentence = []
        max_len = 0
        with open(dict_dir, 'r', encoding = 'utf-8') as f:
            for line in f:
                word = line.rstrip('\n')
                word_dict.add(word)
                max_len = max(max_len, len(word))
        with open(text_dir, 'r', encoding = 'utf-8') as f:
            for line in f:
                sentence.append(line.rstrip('\n'))
        return word_dict, sentence, max_len

    def segment(self):
        """
        前向最大匹配分词
        :return:
        result: 分词结果
        """
        result = []
        with open(out_dir, 'w', encoding = 'utf-8') as out:
            for sen in self.sentence:
                res = []
                left = 0
                right = min(self.len, len(sen))
                while left < len(sen):  # 双指针匹配分词
                    while right > left:
                        if left+1 == right or sen[left:right] in self.word_dict:  # 单字或成功匹配，则加入结果中，否则收缩右指针
                            res.append(sen[left:right])
                            left = right
                            right = min(self.len + left, len(sen))
                        else:
                            right -= 1
                out.write(" ".join(res)+'\n')   # 输出结果
                result.extend(res)
        self.out_words = result
        return result

    def evaluate(self):
        """
        评价函数：根据 正确率/召回率/F值
        :return:
        num_correct: 正确识别的词数
        num_out：识别出的个体总数
        num_test：测试集中的个体总数
        precision：精确度
        recall：召回度
        f1_score：F值
        """
        num_correct = 0  # 正确的词数
        num_out = len(self.out_words)  # 识别出的个体总数
        num_test = len(self.correct_words)  # 测试集中的个体总数
        i = 0
        j = 0
        while i < len(self.out_words) and j < len(self.correct_words):
            # 匹配到正确单词 都向后移一位
            if self.out_words[i] == self.correct_words[j]:
                num_correct += 1
                i += 1
                j += 1
            else:
                offset_i = offset_j = 1
                while i + offset_i < len(self.out_words):
                    offset_j = 1
                    while j + offset_j < len(self.correct_words):
                        if "".join(self.out_words[i:i + offset_i]) == "".join(self.correct_words[j:j + offset_j]):
                            break
                        offset_j += 1
                    if "".join(self.out_words[i:i + offset_i]) == "".join(self.correct_words[j:j + offset_j]):
                        break
                    offset_i += 1
                i += offset_i
                j += offset_j

        precision = num_correct / num_out
        recall = num_correct / num_test
        f1_score = (precision * recall * 2) / (precision + recall)
        print("正确识别的个体总数：{:d}，\n识别出的个体总数：{:d}，\n测试集中存在的个体总数：{:d}，\n正确率：{:f}，\n召回率：{:f}，\nF值：{:f}".format(num_correct, num_out, num_test, precision, recall, f1_score))
        return num_correct, num_out, num_test, precision, recall, f1_score



