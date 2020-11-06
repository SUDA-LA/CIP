import numpy as np
class dataset:
    def __init__(self,filename):
        self.sentences=[]#所有的句子的集合:每个句子是词，词性的元组构成的集合
        self.word=[]#没有重复词的词列表
        self.tag=[]#没有重复词性的词性列表
        self.tag_id={}#词性与id对应的字典
        self.every_sentence=[]#列表中的每一项是一个分好词的句子，每一项都用一个列表表示，列表里放了一个个词
        self.gold_tag=[]#与每个句子中的词对应的词性，每一项也是一个列表
        self.word_total_num=0#总的词的数目
        self.word_num=0#不重复的词的数目
        self.sentence_num=0#总的句子数目
        self.tag_num=0#不重复的词性数目
        self.load_sentence(filename)
        self.filename=filename

    def load_sentence(self,filename):
        """
        对数据集中的数据按照我们需要的方式进行存储
        :param filename: 数据集名称
        :return:
        """
        f = open(filename, "r", encoding="utf-8")
        sentence=[]
        word_set=set()
        tag_set=set()
        for i in f:
            if (len(i) > 1):
                temp_tag = i.split()[3]  #词性
                temp_word = i.split()[1]  #单词
                word_set.add(temp_word)
                tag_set.add(temp_tag)
                sentence.append((temp_word,temp_tag))
                self.word_total_num+=1
            else:
                self.sentences.append(sentence)
                self.gold_tag.append([x[1] for x in sentence])
                self.every_sentence.append([x[0] for x in sentence])
                sentence=[]
        f.close()
        self.word=sorted(list(word_set))
        self.tag=sorted(list(tag_set))
        self.tag_id= {tag:id for id,tag in enumerate(self.tag)}
        self.sentence_num = len(self.sentences)
        self.word_num=len(self.word)
        self.tag_num=len(self.tag)
        print("数据集%s中共有%d个句子，%d个词,其中不重复的词数目为%d个，不重复的词性数目为%d个"% (filename, self.sentence_num, self.word_total_num,self.word_num,self.tag_num))




