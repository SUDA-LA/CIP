import datetime
import random
import numpy as np
from scipy.special import logsumexp
from collections import defaultdict



class dataset:
    def __init__(self, filename):
        f = open(filename, "r", encoding="utf-8")
        self.sentences = []  # 所有的句子的集合
        self.tags = []  # 每个句子对应的词性序列
        sentence = []
        tag = []
        word_num = 0
        for i in f:
            if (len(i) > 1):
                temp_tag = i.split()[3]  # 词性
                temp_word = i.split()[1]  # 单词
                sentence.append(temp_word)
                tag.append(temp_tag)
                word_num += 1
            else:
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
        f.close()
        sentence_count = len(self.sentences)
        print("数据集%s中共有%d个句子，%d个词！" % (filename, sentence_count, word_num))

    def shuffle(self):
        temp=[(s,t) for s,t in zip(self.sentences,self.tags)]
        random.shuffle(temp)
        self.sentences=[]
        self.tags=[]
        for s, t in temp:
            self.sentences.append(s)
            self.tags.append(t)

class Log_Linear_Model_partial:
    def __init__(self,train_data,dev_data,test_data=None):
        self.train_set=dataset(train_data)
        self.dev_set=dataset(dev_data)
        if(test_data!=None):
            self.test_set=dataset(test_data)
        else:
            self.test_set=None
        self.feature_space={}
        self.tag_list=[]

    def create_feature_template(self, sentence, position):
        temp_template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if (position == 0):
            pre_word = "$$"
            pre_word_last_word = "$"
        else:
            pre_word = sentence[position - 1]
            pre_word_last_word = pre_word[-1]
        if (position == len(sentence) - 1):
            next_word = "##"
            next_word_first_char = "#"
        else:
            next_word = sentence[position + 1]
            next_word_first_char = next_word[0]
        temp_template.append("02:" + cur_word)
        temp_template.append("03:" + pre_word)
        temp_template.append("04:" + next_word)
        temp_template.append("05:" + cur_word + "-" + pre_word_last_word)
        temp_template.append("06:" + cur_word + "-" + next_word_first_char)
        temp_template.append("07:" + cur_word_first_char)
        temp_template.append("08:" + cur_word_last_char)
        for i in range(1, len(cur_word) - 1):
            temp_template.append("09:" + cur_word[i])
            temp_template.append("10:" + cur_word_first_char + "-" + cur_word[i])
            temp_template.append("11:" + cur_word_last_char + "-" + cur_word[i])
            if (cur_word[i] == cur_word[i + 1]):
                temp_template.append("13:" + cur_word[i] + '-' + 'consecutive')
        if (len(cur_word) > 1 and cur_word[0] == cur_word[1]):
            temp_template.append("13:" + cur_word[0] + '-' + 'consecutive')
        if (len(cur_word) == 1):
            temp_template.append("12:" + cur_word + "-" + pre_word_last_word + "-" + next_word_first_char)
        for i in range(0, 4):
            if i > len(cur_word) - 1:
                break
            temp_template.append("14:" + cur_word[0:i + 1])
            temp_template.append("15:" + cur_word[-(i + 1)::])
        return temp_template


    def create_feature_space(self):
        all_sentences=self.train_set.sentences
        all_tags=self.train_set.tags
        for i in range(len(all_sentences)):
            temp_sentence=all_sentences[i]
            temp_tag=all_tags[i]
            for j in range(len(temp_sentence)):
                temp_template=self.create_feature_template(temp_sentence,j)
                for k in temp_template:
                    if(k not in self.feature_space):
                        self.feature_space[k]=len(self.feature_space)
            for tag in temp_tag:
                if(tag not in self.tag_list):
                    self.tag_list.append(tag)
        print("整个特征空间总共有%d个特征"%len(self.feature_space))
        self.tag_list.sort()
        self.tag_dict={t:i for i,t in enumerate(self.tag_list)}
        self.weight=np.zeros((len(self.feature_space),len(self.tag_list)))
        self.g = defaultdict(float)


    def calculate_score(self,features):
        scores = np.array([self.weight[self.feature_space[f]] for f in features if f in self.feature_space])
        return np.sum(scores, axis=0)

    def predict(self,sentence,position):
        feature = self.create_feature_template(sentence, position)
        scores = self.calculate_score(feature)
        return self.tag_list[int(np.argmax(scores))]

    def evaluate(self,data_set):
        all_sentence=data_set.sentences
        all_tag=data_set.tags
        total_num=0
        correct_num=0
        for i in range(len(all_sentence)):
            sentence=all_sentence[i]
            tag=all_tag[i]
            total_num+=len(tag)
            for j in range(len(sentence)):
                predict_tag=self.predict(sentence,j)
                if(predict_tag==tag[j]):
                    correct_num+=1
        return (correct_num,total_num,correct_num/total_num)
    #iterations表示的是迭代次数，max_iterations表示多少次迭代没有提升则退出，eta指的是学习率，batch_size指的是批量次数,C指的是正则化系数，decay指的是学习率的衰退系数，对于这次优化，在正则项方面使用C作为正则项系数
    def SGD_Training(self,iterations,max_iterations,eta,batch_size,C,decay,shuffle=False,regularization=False,Simulated_annealing=False):
        train_sentences=self.train_set.sentences
        train_tags=self.train_set.tags
        max_accuracy_rate=0
        highest_accuracy_iterations=-1
        counter=0
        b=0
        n=len(train_sentences)
        #N=sum([len(sent) for sent in train_sentences])
        N=100000
        num=1
        learn_rate=eta
        print("eta(学习率) = %f"%(eta))
        if(regularization):
            print("使用正则项对模型进行优化！")
        if(Simulated_annealing):
            print("使用模拟退火算法对模型进行优化！")
        for m in range(iterations):
            print("第%d轮迭代:"%(m+1))
            starttime = datetime.datetime.now()
            if(shuffle==True):
                print("在这一轮迭代中打乱所有的训练集数据！")
                self.train_set.shuffle()
                train_sentences = self.train_set.sentences
                train_tags = self.train_set.tags
            for j in range(len(train_sentences)):
                sentence=train_sentences[j]
                tags=train_tags[j]
                for i in range(len(sentence)):
                    #实现算法第7行这一公式
                    correct_tag=tags[i]
                    feature_template=self.create_feature_template(sentence,i)
                    all_score_lst=self.calculate_score(feature_template)
                    sum_score=logsumexp(all_score_lst)#先求它的exp，然后求和，再求和的log
                    temp_result=np.exp(all_score_lst-sum_score)
                    for f in feature_template:
                        self.g[self.feature_space[f]]-=temp_result
                        self.g[(self.feature_space[f],self.tag_dict[correct_tag])]+=1
                    b=b+1
                    if(batch_size==b):
                        if(regularization):
                            self.weight*=(1-learn_rate*C)
                        for id, value in self.g.items():
                            self.weight[id] += learn_rate * value
                        if(Simulated_annealing):
                            learn_rate=eta*decay**(num/N)
                        num+=1
                        b=0
                        self.g = defaultdict(float)
            if(b>0):
                if (regularization):
                    self.weight *= (1 - learn_rate * C)
                for id, value in self.g.items():
                    self.weight[id] += learn_rate * value
                if (Simulated_annealing):
                    learn_rate = eta * decay ** (num / N)
                num += 1
                b=0
                self.g = defaultdict(float)
            #评价程序
            train_correct_num, total_num, train_precision = self.evaluate(self.train_set)
            print('train(训练集)准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_set)
            print('dev(开发集)准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))
            if(self.test_set!=None):
                test_correct_num, test_num, test_precision = self.evaluate(self.test_set)
                print('test(测试集)准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))
            if dev_precision > max_accuracy_rate:
                max_accuracy_rate = dev_precision
                highest_accuracy_iterations=m
                counter=0
            else:
                counter+=1
            endtime = datetime.datetime.now()
            print("第%d次迭代所花费的时间为:%sS" % (m+1,endtime-starttime))
            """
           if (train_correct_num == total_num):
              break
           """
            if (counter >= max_iterations):
                break
        print('第%d次迭代对应的开发集预测的准确率最高，最高的准确率为:%f' % (highest_accuracy_iterations+1, max_accuracy_rate))



if __name__ == '__main__':
    m=Log_Linear_Model_partial("train.conll","dev.conll")
    m.create_feature_space()
    m.SGD_Training(50,20,0.5,50,0.0001,0.96,shuffle=False,regularization=False,Simulated_annealing=False)
