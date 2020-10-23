import numpy as np
import datetime
import random
from collections import defaultdict
from scipy.special import logsumexp

class dataset:
    """
    dataset类主要用于对给定文件中的数据进行处理，将每个句子和他们的词性标签相对应，同时包含一个打乱数据集中数据的函数。
    """
    def __init__(self, filename):
        f = open(filename, "r", encoding="utf-8")
        self.sentences = []  # 所有的句子的集合
        self.tags = []  # 每个句子对应的词性序列
        sentence = []
        tag = []
        word_num = 0
        for i in f:
            if (i[0]!=" "and len(i) > 1):
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
        print("数据集%s中共有%d个句子，%d个词！" % (filename.split("/")[-1], sentence_count, word_num))

    def shuffle(self):
        temp=[(s,t) for s,t in zip(self.sentences,self.tags)]
        random.shuffle(temp)
        self.sentences=[]
        self.tags=[]
        for s, t in temp:
            self.sentences.append(s)
            self.tags.append(t)

class CRF_partial:
    def __init__(self,train_data,dev_data,test_data=None):
        """
        使用了特征抽取优化的CRF模型的构造函数
        :param train_data:#训练数据集
        :param dev_data:#开发数据集
        :param test_data:#测试数据集
        """
        self.train_set=dataset(train_data)
        self.dev_set=dataset(dev_data)
        if(test_data!=None):
            self.test_set=dataset(test_data)
        else:
            self.test_set=None
        self.feature_space={}
        self.tag_list=[]
        self.FIR="FIR"

    def create_bigram_feature_template(self,pretag):
        """
        构造基于bigram特征的部分特征模板
        :param pretag: 前一个词的词性标签
        :return:特征模板集合
        """
        template=["01:"+pretag]
        return template

    def create_uigram_feature_template(self,sentence,position):
        """
        构造基于uigram特征的部分特征模板，没有词性信息
        :param sentence: 一个分好词的句子的列表
        :param position:当前词在句子中的位置
        :return:特征模板集合
        """
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


    def create_feature_template(self,sentence,position,pretag):
        """
        构造整个的部分特征模板，包括uigram和bigram
        :param sentence: 一个分好词的句子的列表
        :param position: 当前词在句子中的位置
        :param pretag: 当前词的前一个词的词性标签
        :return: 特征模板集合
        """
        feature_template=self.create_bigram_feature_template(pretag)
        feature_template.extend(self.create_uigram_feature_template(sentence,position))
        return feature_template

    def create_feature_space(self):
        """
        构造整个训练集的部分特征空间
        self.feature_space对应的就是部分特征空间的集合
        还构造了一些后面需要用到的比如权重向量，梯度向量等
        :return:
        """
        all_sentences = self.train_set.sentences
        all_tags = self.train_set.tags
        for i in range(len(all_sentences)):
            temp_sentence = all_sentences[i]
            temp_tag = all_tags[i]
            for j in range(len(temp_sentence)):
                if (j == 0):
                    pretag = self.FIR
                else:
                    pretag = temp_tag[j - 1]
                feature_template = self.create_feature_template(temp_sentence, j, pretag)
                for k in feature_template:
                    if (k not in self.feature_space):
                        self.feature_space[k] = len(self.feature_space)
                if (temp_tag[j] not in self.tag_list):
                    self.tag_list.append(temp_tag[j])
        print("整个特征空间总共有%d个特征" % len(self.feature_space))
        self.tag_list.sort()
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.weight = np.zeros((len(self.feature_space), len(self.tag_dict)))
        self.g = defaultdict(float)
        self.bigram_feature_template = [self.create_bigram_feature_template(pre) for pre in self.tag_list]
        self.bigram_scores = np.array([self.calculate_score(f) for f in self.bigram_feature_template])

    def calculate_score(self,features):
        """
        计算得分，这里的计算得分和不加入特征抽取优化的计算得分是有区别的，这里返回的是一个向量，对应了词性分别为词性列表中每个标签的时候，所对应的部分特征的得分。
        而不加入特征抽取优化计算出来的得分，则就是将当前词标注为该词性所得到的所有特征构成的分数，是一个浮点数。
        :param features: 对应的特征模板
        :return: 词性分别为词性列表中每个标签的时候，所对应的部分特征的得分的集合
        """
        scores = np.array([self.weight[self.feature_space[f]] for f in features if f in self.feature_space])
        return np.sum(scores, axis=0)

    def predict(self,sentence):
        """
        给定一个句子，预测他的词性标签，用到了Viterbi算法
        :param sentence: 一个给定的句子
        :return: 整个句子每个词对应的词性标签的列表
        """
        N=len(sentence)
        useful_score=np.zeros((N,len(self.tag_list)))
        path=np.zeros((N,len(self.tag_list)),dtype="int")
        path[0]=-1
        first_feature_template_lst=self.create_feature_template(sentence,0,self.FIR)#句子的第一个词标注为每一个词性所构成的特征向量的集合
        useful_score[0]=self.calculate_score(first_feature_template_lst)#句子的第一个词标注为每一个词性对应的得分
        bigram_score_all_lst=[self.calculate_score(features) for features in self.bigram_feature_template]
        for i in range(1,N):
            uigram_template_feature=self.create_uigram_feature_template(sentence,i)
            uigram_score=self.calculate_score(uigram_template_feature)
            temp_score=[useful_score[i-1][j]+bigram_score_all_lst[j]+uigram_score for j in range(len(self.tag_list))]
            path[i]=np.argmax(temp_score,axis=0)
            useful_score[i]=np.max(temp_score,axis=0)

        last=int(np.argmax(useful_score[-1]))
        last_tag=self.tag_list[last]
        predict_tag_lst=[last_tag]
        T=len(sentence)-1
        for i in range(T,0,-1):
            last=path[i][last]
            predict_tag_lst.insert(0,self.tag_list[last])
        return predict_tag_lst

    def evaluate(self,data_set):
        """
        评价函数
        :param data_set: 需要预测评价的数据集
        :return: 正确预测的词的数目，所有词的数目，预测的准确率
        """
        all_sentence=data_set.sentences
        all_tag=data_set.tags
        total_num=0
        correct_num=0
        for i in range(len(all_sentence)):
            sentence=all_sentence[i]
            tag=all_tag[i]
            total_num+=len(tag)
            predict_tag_lst=self.predict(sentence)
            for j in range(len(tag)):
                if(predict_tag_lst[j]==tag[j]):
                    correct_num+=1
        return (correct_num,total_num,correct_num/total_num)

    def forward_algorithm(self,sentence):
        """
        forward算法，计算前向得分
        :param sentence: 分好词的一个句子列表
        :return: 返回该句子每个词和他的所有可能的词性的前向得分矩阵
        """
        path_score=np.zeros((len(sentence),len(self.tag_list)))
        template=self.create_feature_template(sentence,0,self.FIR)
        path_score[0]=self.calculate_score(template)
        for i in range (1,len(sentence)):
            uigram_template=self.create_uigram_feature_template(sentence,i)
            uigram_score=self.calculate_score(uigram_template)
            total_score=self.bigram_scores+uigram_score
            path_score[i]=logsumexp(total_score.T+path_score[i-1],axis=1)
        return path_score

    def backward_algorithm(self,sentence):
        """
        backward算法，计算后向得分
        :param sentence: 分好词的一个句子列表
        :return: 返回该句子每个词和他的所有可能的词性的后向得分矩阵
        """
        path_score = np.zeros((len(sentence),len(self.tag_list)))
        for i in reversed(range(len(sentence) - 1)):
            uigram_template=self.create_uigram_feature_template(sentence,i+1)
            uigram_score=self.calculate_score(uigram_template)
            total_score=self.bigram_scores+uigram_score
            path_score[i]=logsumexp(total_score+path_score[i+1],axis=1)
        return path_score

    def update_gradient(self,sentence,tags):
        """
        梯度更新算法
        :param sentence:分好词的一个句子列表
        :param tags:这个句子每个词对应的标签列表
        :return:
        """
        for i in range(len(sentence)):
            uigram_template=self.create_uigram_feature_template(sentence,i)
            if(i==0):
                bigram_template=self.create_bigram_feature_template(self.FIR)
            else:
                #bigram_template=self.create_bigram_feature_template(tags[i-1])
                bigram_template=self.bigram_feature_template[self.tag_dict[tags[i-1]]]
            curtag=tags[i]
            for f in uigram_template:
                if(f in self.feature_space):
                    self.g[(self.feature_space[f],self.tag_dict[curtag])]+=1
            for f in bigram_template:
                if(f in self.feature_space):
                    self.g[(self.feature_space[f],self.tag_dict[curtag])]+=1
        forward_score = self.forward_algorithm(sentence)
        backward_score = self.backward_algorithm(sentence)
        zs = logsumexp(forward_score[-1])  # 分母
        template=self.create_feature_template(sentence, 0, self.FIR)
        p=np.exp(self.calculate_score(template)+backward_score[0] - zs)
        id = [self.feature_space[f] for f in template if f in self.feature_space]
        for f in id:
            self.g[f]-=p
        for i in range(1,len(sentence)):
            uigram_template=self.create_uigram_feature_template(sentence,i)
            total_score=self.calculate_score(uigram_template)+self.bigram_scores
            p_lst=np.exp(total_score+forward_score[i-1][:,None]+backward_score[i]-zs)
            uigram_id=[self.feature_space[f] for f in uigram_template if f in self.feature_space]
            for bigram_template,p in zip(self.bigram_feature_template, p_lst):
                bigram_id = [self.feature_space[f] for f in bigram_template if f in self.feature_space]
                all_id = uigram_id + bigram_id
                for f in all_id:
                    self.g[f]-=p

    def SGD_Training(self, iterations, max_iterations, eta, batch_size, C, decay, shuffle, regularization,Simulated_annealing):
        """
        SGD训练，利用梯度下降的方法对特征权重进行训练的算法
        :param iterations: 最大的迭代次数
        :param max_iterations: 迭代多少次没有提升则退出
        :param eta: 学习率
        :param batch_size:批次大小，多少次用梯度向量更新权重向量。
        :param C:正则化系数
        :param decay:学习率的衰退系数
        :param shuffle:是否打乱顺序
        :param regularization:是否使用正则化
        :param Simulated_annealing:是否使用模拟退火
        :return:
        """
        train_sentences = self.train_set.sentences
        train_tags = self.train_set.tags
        max_accuracy_rate = 0
        highest_accuracy_iterations = -1
        counter = 0
        N = 100000
        num = 1
        learn_rate = eta
        print("eta(学习率) = %f" % (eta))
        if (regularization):
            print("使用正则项对模型进行优化！")
        if (Simulated_annealing):
            print("使用模拟退火算法对模型进行优化！")
        for m in range(iterations):
            b = 0
            print("第%d轮迭代:" % (m + 1))
            starttime = datetime.datetime.now()
            if (shuffle == True):
                print("在这一轮迭代中打乱所有的训练集数据！")
                self.train_set.shuffle()
                train_sentences = self.train_set.sentences
                train_tags = self.train_set.tags
            for j in range(len(train_sentences)):
                sentence = train_sentences[j]
                tags = train_tags[j]
                self.update_gradient(sentence,tags)
                b=b+1
                if (batch_size==b):
                    if (regularization):
                        self.weight *= (1 - learn_rate * C)
                    for id, value in self.g.items():
                        self.weight[id] += learn_rate * value
                    if (Simulated_annealing):
                        learn_rate = eta * decay ** (num / N)
                    num+=1
                    b=0
                    self.g = defaultdict(float)
                    self.bigram_scores = np.array([self.calculate_score(f) for f in self.bigram_feature_template])#一定要写
            if (b > 0):
                if (regularization):
                    self.weight *= (1 - learn_rate * C)
                for id, value in self.g.items():
                    self.weight[id] += learn_rate * value
                if (Simulated_annealing):
                    learn_rate = eta * decay ** (num / N)
                num += 1
                b = 0
                self.g = defaultdict(float)
                self.bigram_scores = np.array([self.calculate_score(f) for f in self.bigram_feature_template])#一定要写
            # 评价程序
            train_correct_num, total_num, train_precision = self.evaluate(self.train_set)
            print('train(训练集)准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_set)
            print('dev(开发集)准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))
            if (self.test_set != None):
                test_correct_num, test_num, test_precision = self.evaluate(self.test_set)
                print('test(测试集)准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))
            if dev_precision > max_accuracy_rate:
                max_accuracy_rate = dev_precision
                highest_accuracy_iterations = m
                counter = 0
            else:
                counter += 1
            endtime = datetime.datetime.now()
            print("第%d次迭代所花费的时间为:%sS" % (m + 1, endtime - starttime))
            """
           if (train_correct_num == total_num):
              break
           """
            if (counter >= max_iterations):
                break
        print('第%d次迭代对应的开发集预测的准确率最高，最高的准确率为:%f' % (highest_accuracy_iterations+1, max_accuracy_rate))














