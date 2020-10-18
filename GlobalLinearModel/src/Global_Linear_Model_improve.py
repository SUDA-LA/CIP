import numpy as np
import datetime
import random

class dataset:
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

class Global_Linear_Model_improve:
    def __init__(self,train_data,dev_data,test_data=None):
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
        template=["01:"+pretag]
        return template

    def create_uigram_feature_template(self,sentence,position):
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
        feature_template=self.create_bigram_feature_template(pretag)
        feature_template.extend(self.create_uigram_feature_template(sentence,position))
        return feature_template

    def create_feature_space(self):
        all_sentences=self.train_set.sentences
        all_tags=self.train_set.tags
        for i in range(len(all_sentences)):
            temp_sentence=all_sentences[i]
            temp_tag=all_tags[i]
            for j in range(len(temp_sentence)):
                if(j==0):
                    pretag=self.FIR
                else:
                    pretag=temp_tag[j-1]
                feature_template=self.create_feature_template(temp_sentence,j,pretag)
                for k in feature_template:
                    if(k not in self.feature_space):
                        self.feature_space[k]=len(self.feature_space)
                if(temp_tag[j] not in self.tag_list):
                    self.tag_list.append(temp_tag[j])
        print("整个特征空间总共有%d个特征" % len(self.feature_space))
        self.tag_list.sort()
        self.tag_dict = {t: i for i, t in enumerate(self.tag_list)}
        self.idtag_dict = {i:t for i, t in enumerate(self.tag_list)}
        self.weight=np.zeros((len(self.feature_space),len(self.tag_dict)),dtype="int32")
        self.v=np.zeros((len(self.feature_space),len(self.tag_dict)), dtype="int32")
        self.update_time = np.zeros((len(self.feature_space),len(self.tag_dict)), dtype="int")
        #所有可能的当前词性和所有可能的前一个词性所构成的特征向量的列表，行表示当前词性，列表示它的前一个词性
        self.bigram_feature_template_all_lst=[self.create_bigram_feature_template(pre) for pre in self.tag_list]

    def calculate_score(self,features,averaged_perceptron=False):
        score=[]
        for f in features:
            if(f in self.feature_space):
                if(averaged_perceptron==True):
                    score.append(self.v[self.feature_space[f]])
                else:
                    score.append(self.weight[self.feature_space[f]])
        return np.sum(score, axis=0)

    def predict(self,sentence,averaged_perceptron):
        N=len(sentence)
        useful_score=np.zeros((N,len(self.tag_list)))
        path=np.zeros((N,len(self.tag_list)),dtype="int")
        path[0]=-1
        first_feature_template_lst=self.create_feature_template(sentence,0,self.FIR)#句子的第一个词标注为每一个词性所构成的特征向量的集合
        useful_score[0]=self.calculate_score(first_feature_template_lst,averaged_perceptron)#句子的第一个词标注为每一个词性对应的得分
        bigram_score_all_lst=[self.calculate_score(features,averaged_perceptron) for features in self.bigram_feature_template_all_lst]
        for i in range(1,N):
            uigram_template_feature=self.create_uigram_feature_template(sentence,i)
            uigram_score=self.calculate_score(uigram_template_feature,averaged_perceptron)
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

    def evaluate(self,data_set,average_perceptron):
        all_sentence=data_set.sentences
        all_tag=data_set.tags
        total_num=0
        correct_num=0
        for i in range(len(all_sentence)):
            sentence=all_sentence[i]
            tag=all_tag[i]
            total_num+=len(tag)
            predict_tag_lst=self.predict(sentence,average_perceptron)
            for j in range(len(tag)):
                if(predict_tag_lst[j]==tag[j]):
                    correct_num+=1
        return (correct_num,total_num,correct_num/total_num)

    def update(self,current_time,index1,index2,last_w):
        last_update=self.update_time[(index1,index2)]
        if(current_time!=last_update):
            self.update_time[index1][index2]=current_time
            self.v[index1][index2]+=(current_time-last_update-1)*last_w+self.weight[index1][index2]


    def Online_Training(self,iterations,max_iterations,average_perceptron,shuffle):
        train_sentences=self.train_set.sentences
        train_tags=self.train_set.tags
        max_accuracy_rate=0
        highest_accuracy_iterations=-1
        counter=0
        update=0
        if(average_perceptron==True):
            print("在本次训练预测过程中使用self.v来代替self.weight对结果进行预测")
        else:
            print("在本次训练预测过程中使用self.weight对结果进行预测")
        for m in range(iterations):
            print("第%d轮迭代:" % (m+1))
            starttime=datetime.datetime.now()
            if(shuffle==True):
                print("在这一轮迭代中打乱所有的训练集数据！")
                self.train_set.shuffle()
                train_sentences = self.train_set.sentences
                train_tags = self.train_set.tags
            for j in range(len(train_sentences)):
                sentence=train_sentences[j]
                correct_tag_lst=train_tags[j]
                predict_tag_lst=self.predict(sentence,False)
                if predict_tag_lst != correct_tag_lst:
                    update+=1
                    for i in range(len(correct_tag_lst)):
                        unigram_feature = self.create_uigram_feature_template(sentence,i)
                        if(i==0):
                            correct_feature_template=self.create_bigram_feature_template(self.FIR)
                            predict_feature_template=self.create_bigram_feature_template(self.FIR)
                        else:
                            correct_pre_tag=correct_tag_lst[i-1]
                            predict_pre_tag=predict_tag_lst[i-1]
                            correct_feature_template=self.bigram_feature_template_all_lst[self.tag_dict[correct_pre_tag]]
                            predict_feature_template=self.bigram_feature_template_all_lst[self.tag_dict[predict_pre_tag]]
                        for f in unigram_feature:
                            if(f in self.feature_space):
                                index1=self.feature_space[f]
                                index2=self.tag_dict[correct_tag_lst[i]]
                                last_w=self.weight[index1][index2]
                                self.weight[index1][index2]+=1
                                self.update(update,index1,index2,last_w)
                                index1 = self.feature_space[f]
                                index2 = self.tag_dict[predict_tag_lst[i]]
                                last_w = self.weight[index1][index2]
                                self.weight[index1][index2]-=1
                                self.update(update, index1, index2, last_w)
                        for f in correct_feature_template:
                            if(f in self.feature_space):
                                index1=self.feature_space[f]
                                index2=self.tag_dict[correct_tag_lst[i]]
                                last_w=self.weight[index1][index2]
                                self.weight[index1][index2]+=1
                                self.update(update,index1,index2,last_w)
                        for f in predict_feature_template:
                            if (f in self.feature_space):
                                index1 = self.feature_space[f]
                                index2 = self.tag_dict[predict_tag_lst[i]]
                                last_w = self.weight[index1][index2]
                                self.weight[index1][index2]-=1
                                self.update(update, index1, index2, last_w)
            if (average_perceptron):
                cur_update_time = update
                for k in range(len(self.update_time)):
                    for kk in range(len(self.tag_dict)):
                        last_w=self.weight[k][kk]
                        self.update(cur_update_time,k,kk,last_w)
            #评价程序
            train_correct_num,total_num,train_precision=self.evaluate(self.train_set,average_perceptron)
            print('train(训练集)准确率：%d / %d = %f'% (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_set, average_perceptron)
            print('dev(开发集)准确率：%d / %d = %f'%(dev_correct_num, dev_num, dev_precision))
            if(self.test_set!=None):
                test_correct_num, test_num, test_precision = self.evaluate(self.test_set, average_perceptron)
                print('test(测试集)准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))
            if dev_precision>max_accuracy_rate:
                max_accuracy_rate=dev_precision
                highest_accuracy_iterations=m
                counter=0
            else:
                counter+=1
            endtime = datetime.datetime.now()
            print("第%d次迭代所花费的时间为:%sS"%(m+1,endtime-starttime))
            if(counter>=max_iterations):
                break
        print('第%d次迭代对应的开发集预测的准确率最高，最高的准确率为:%f'%(highest_accuracy_iterations+1,max_accuracy_rate))

if __name__ == '__main__':
    m=Global_Linear_Model("train.conll","dev.conll")
    m.create_feature_space()
    m.Online_Training(50,10,shuffle=True,average_perceptron=True)





