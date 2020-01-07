# -*- coding: utf-8 -*-
from xpinyin import Pinyin

if __name__ == "__main__":
    '''
    把conll文件内容读到dict中，以首汉字为key，词列表为value，按拼音顺序写入文件
    格式为 [首汉字1]:[以该汉字为首的词语1] [词语2]......
           [首汉字2]:[以该汉字为首的词语1] [词语2]......
    '''
    fin = open("./data/data.conll", "r")
    fout = open("./data/word_dict.txt", "w")
    dict = {}
    maxwordlen = 0  # 用来获取词的最大长度
    while True:
        line = fin.readline()
        if not line:
            break
        if line != "\n":
            L = line.split()  # 按空格分割字符串获取八个值
            firstword = L[1].decode("utf-8")[0:1].encode("utf-8")  # 获取第一个汉字
            wordlen = len(L[1].decode("utf-8"))  # 获取词的长度
            if wordlen > maxwordlen:
                maxwordlen = wordlen
            # 如果该首汉字未出现过，就新插入一个空列表
            if not dict.has_key(firstword):
                dict[firstword] = []
            # 如果不是重复词就放入列表中
            if L[1] not in dict[firstword]:
                dict[firstword].append(L[1])
    print "最大单词长度为：%d" % maxwordlen

    '''
    用于检验字典是否正确
    for firstword,words in dict.items():
        print firstword+":",
        for word in words:
            print  word+" ",
        print "\n",
    '''

    p = Pinyin()
    # 对每个首汉字对应的词列表按拼音顺序排序
    for val in dict.values():
        val.sort(key=lambda x: p.get_pinyin(unicode(x, 'utf-8'), ""))
    # 对整个dict按首汉字拼音排序 并写入文件
    for item in sorted(dict.items(), key=lambda x: p.get_pinyin(unicode(x[0], 'utf-8'), "")):
        fout.write(item[0])
        fout.write(":")
        for word in item[1]:
            fout.write(word + " ")
        fout.write("\n")

    fin.close()
    fout.close()
