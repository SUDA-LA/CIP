# -*- coding: utf-8 -*-


def create_dict(fin):
    '''
    用于初始化词典函数
    返回一个dict和词的最大长度
    dict以首汉字为key，单词列表为value
    '''
    dict = {}
    maxlength = 0;
    while True:
        line = fin.readline()
        if not line:
            break
        temp = line.split(":")
        if not dict.has_key(temp[0]):
            dict[temp[0]] = []
        # 注意舍去最后的换行符
        for word in temp[1][:-1].split():
            dict[temp[0]].append(word)
            length = len(word.decode("utf-8"))
            if length > maxlength:
                maxlength = length
    return dict, maxlength


def word_split(string):
    '''
    正向最大匹配算法，参数为要分词的字符串
    返回一个列表，每个元素即为分好的词
    '''
    res = []
    text = string.decode("utf-8")
    while len(text) > 0:
        length = MAX_LENGTH  # 单词最大长度
        if len(text) < length:  # 若字符串本身小于最大长度，就取字符串长度
            length = len(text)
        tryword = text[0:0 + length]  # 截取首个最长的单词
        firstword = text[0:1]  # 截取首汉字

        # 判断该字符串是否在词典中出现
        while not dict.has_key(firstword.encode("utf-8")) or \
                tryword.encode("utf-8") not in dict[firstword.encode("utf-8")]:
            # 如果只剩一个字了就直接退出
            if (len(tryword) == 1):
                break
            # 每次长度-1
            tryword = tryword[0:len(tryword) - 1]
        # 插入列表中
        res.append(tryword.encode("utf-8"))
        # 截取后面的字符串重新开始
        text = text[len(tryword):]
    return res


if __name__ == "__main__":
    fin1 = open("./data/data.txt", "r")
    fin2 = open("./data/word_dict.txt", "r")
    fout = open("./data/out.txt", "w")
    dict, MAX_LENGTH = create_dict(fin2)
    print "初始化词典完成，最大长度为%d" % MAX_LENGTH
    '''
    用于检验词典是否正确
    for firstword,words in dict.items():
        print firstword+":",
        for word in words:
            print  word+" ",
        print "\n",
    '''

    while True:
        line = fin1.readline()
        if not line:
            break
        res = word_split(line[:-1])  # 注意舍去换行符
        for word in res:
            fout.write(word + " ")
        fout.write("\n")

    fin1.close()
    fin2.close()
    fout.close()
