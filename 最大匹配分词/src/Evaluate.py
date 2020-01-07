# -*- coding: utf-8 -*-

def get_string(list):
    res = ""
    for item in list:
        res = res + item
    return res


def evaluate(out_words, correct_words):
    # 链接法
    num_correct = 0  # 正确的词数
    num_out = len(out_words)  # 识别出的总体个数
    num_test = len(correct_words)  # 测试集中的个体总数
    i = 0
    j = 0
    while i < len(out_words) and j < len(correct_words):
        # 匹配到正确单词 都向后移一位
        if out_words[i] == correct_words[j]:
            num_correct += 1
            i += 1
            j += 1
        else:
            offset_i = offset_j = 1
            while i + offset_i < len(out_words):
                offset_j = 1
                while j + offset_j < len(correct_words):
                    if get_string(out_words[i:i + offset_i]) == get_string(correct_words[j:j + offset_j]):
                        break
                    offset_j += 1
                if get_string(out_words[i:i + offset_i]) == get_string(correct_words[j:j + offset_j]):
                    break
                offset_i += 1
            i += offset_i
            j += offset_j

    return num_correct, num_out, num_test


if __name__ == "__main__":
    fin_out = open("./data/out.txt", "r")
    fin = open("./data/data.conll", "r")
    num_correct = 0  # 正确的词数
    num_out = 0  # 识别出的总体个数
    num_test = 0  # 测试集中的个体总数
    while True:
        # 读取out文件中的一行，并且按空格分割，把每个词依次放在列表中
        line_out = fin_out.readline()
        if not line_out:
            break
        out_words = line_out[:-1].split()
        correct_words = []
        # 读取conll文件中的一个句子，把正确的词依次放在列表中
        while True:
            line = fin.readline()
            if line == "\n":
                break
            correct_words.append(line.split()[1])

        res = evaluate(out_words, correct_words)
        num_correct += res[0]
        num_out += res[1]
        num_test += res[2]

    print "正确识别的词数：%d" % num_correct
    print "识别出的总体个数：%d" % num_out
    print "测试集中的总体个数：%d" % num_test
    Precision = num_correct / float(num_out)
    Recall = num_correct / float(num_test)
    F = Precision * Recall * 2 / (Precision + Recall)
    print "正确率：%.5f" % Precision
    print "召回率：%.5f" % Recall
    print "F值：%.5f" % F

    fin_out.close()
    fin.close()
