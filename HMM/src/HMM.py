import numpy as np
import datetime

class data_set:
    def __init__(self,filename):
        """
        dataset类，主要作用是讲文本形式的数据集转化为我们所需要的列表形式的数据集存储在内存中，方便后面进行操作和调用
        :param filename: 文本名称
        """
        f=open(filename,"r",encoding="utf-8")
        self.sentences=[]#所有的句子的集合
        self.tags=[]#每个句子对应的词性序列
        sentence=[]
        tag=[]
        word_num=0
        for i in f:
            if(i[0]!=" " and len(i)>1):
                temp_tag=i.split()[3]#词性
                temp_word=i.split()[1]#单词
                sentence.append(temp_word)
                tag.append(temp_tag)
                word_num+=1
            else:
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence=[]
                tag=[]
        f.close()
        sentence_count=len(self.sentences)
        print("数据集%s中共有%d个句子，%d个词！" %(filename,sentence_count,word_num))


class HMM:
    def __init__(self,train_data,alpha):
        """
        构造函数，对所需要的各种集合字典进行初始化构造
        :param train_data: 训练集数据
        :param alpha:平滑参数
        """
        self.train_set=data_set(train_data)#训练集生成的数据对象
        self.train_sentences=self.train_set.sentences#训练集中的句子的集合
        self.train_tags=self.train_set.tags#训练集中的词性的集合
        self.sentence_count=len(self.train_sentences)#训练集中句子的个数
        self.tags_dict={}#所有词性以及每个词性出现次数的字典
        self.word_dict={}#所有词以及每个词出现次数的字典
        self.tag_word_dict={}#词性发射到词以及对应出现次数的的字典
        self.first_tag_dict={}#各词性在序列首结点以及对应出现次数的字典
        self.tag_tag_dict={}#两个连续出现的词性和他们在训练集中出现的词数构成的字典
        self.alpha=alpha
        self.UNK="UNK"


    def achieve_train_data_set(self):
        """
        获取后面训练函数所需要的数据集和一些有用的数据
        :return:
        """
        for i in range(self.sentence_count):
            sentence=self.train_sentences[i]
            tag=self.train_tags[i]
            first_tag=tag[0]
            if(first_tag not in self.first_tag_dict):
                self.first_tag_dict[first_tag]=1
            else:
                self.first_tag_dict[first_tag]+=1
            for j in range(len(tag)-1):
                tag_tag=tag[j]+"*"+tag[j+1]
                if(tag_tag not in self.tag_tag_dict):
                    self.tag_tag_dict[tag_tag]=1
                else:
                    self.tag_tag_dict[tag_tag]+=1
                temp_tag=tag[j]
                temp_word=sentence[j]
                tag_word=temp_tag+"*"+temp_word
                if(tag_word not in self.tag_word_dict):
                    self.tag_word_dict[tag_word]=1
                else:
                    self.tag_word_dict[tag_word]+=1
                if(temp_tag not in self.tags_dict):
                    self.tags_dict[temp_tag]=1
                else:
                    self.tags_dict[temp_tag]+=1
                if(temp_word not in self.word_dict):
                    self.word_dict[temp_word]=1
                else:
                    self.word_dict[temp_word]+=1
            last_tag=tag[-1]
            last_word=sentence[-1]
            last_tag_word=last_tag+"*"+last_word
            if (last_tag_word not in self.tag_word_dict):
                self.tag_word_dict[last_tag_word] = 1
            else:
                self.tag_word_dict[last_tag_word] += 1
            if (last_tag not in self.tags_dict):
                self.tags_dict[last_tag] = 1
            else:
                self.tags_dict[last_tag] += 1
            if (last_word not in self.word_dict):
                self.word_dict[last_word] = 1
            else:
                self.word_dict[last_word]+=1
        self.word_dict[self.UNK]=1

        tag_lst=self.tags_dict.keys()
        self.tag_count_dic={i: t for i, t in enumerate(tag_lst)}
        word_lst=self.word_dict.keys()
        self.word_count_dic={i: t for i, t in enumerate(word_lst)}

    def training(self,alpha):
        """
        是一个训练函数，主要用于根据训练集训练出后面预测所需要的初始矩阵，发射矩阵和转移矩阵。
        :param alpha: 平滑参数
        :return:
        """
        #1.建立初始矩阵
        self.initial_matrix=np.zeros(len(self.tags_dict))
        initial_all_count=sum(list(self.first_tag_dict.values()))
        for i in range(len(self.tags_dict)):
            tag=self.tag_count_dic[i]
            if(tag in self.first_tag_dict):
                self.initial_matrix[i]=(self.first_tag_dict[tag]+alpha)/(initial_all_count+alpha*len(self.tags_dict))
            else:
                self.initial_matrix[i]=(alpha/(initial_all_count+alpha*len(self.tags_dict)))

        #2.建立发射矩阵
        self.launch_matrix=np.zeros((len(self.tags_dict),len(self.word_dict)))
        for i in range(len(self.tags_dict)):
            tag=self.tag_count_dic[i]
            tag_word_all_count = self.tags_dict[tag]
            for j in range(len(self.word_dict)):
                word=self.word_count_dic[j]
                composite_temp=tag+"*"+word
                if(composite_temp in self.tag_word_dict):
                    self.launch_matrix[i][j]=(self.tag_word_dict[composite_temp]+alpha)/(tag_word_all_count+alpha*len(self.word_dict))
                else:
                    self.launch_matrix[i][j]=alpha/(tag_word_all_count+alpha*len(self.word_dict))

        #3.建立转移矩阵
        self.transfer_matrix=np.zeros((len(self.tags_dict),len(self.tags_dict)))
        for i in range(len(self.tags_dict)):
            tag1=self.tag_count_dic[i]
            tag_tag_all_count=self.tags_dict[tag1]
            for j in range(len(self.tags_dict)):
                tag2=self.tag_count_dic[j]
                composite_temp=tag1+"*"+tag2
                if(composite_temp in self.tag_tag_dict):
                    self.transfer_matrix[i][j]=(self.tag_tag_dict[composite_temp]+alpha)/(tag_tag_all_count+alpha*len(self.tags_dict))
                else:
                    self.transfer_matrix[i][j]=alpha/(tag_tag_all_count+alpha*len(self.tags_dict))

    #输入一个句子，预测结果序列
    def predict(self,sentence,alpha):
        """
        预测函数，实现输入一个句子，输出这个句子根据模型预测得出的完整的词性序列
        :param sentence:一个中文分过词的句子
        :param alpha: 平滑参数
        :return: 一个词性序列列表，列表中的每一项对应了输入句子中的每个词
        """
        score=np.zeros((len(sentence),len(self.tags_dict)))
        path=np.zeros((len(sentence),len(self.tags_dict)),int)
        initial_matrix=np.log(self.initial_matrix)
        launch_matrix=np.log(self.launch_matrix)
        transfer_matrix = np.log(self.transfer_matrix)
        for i in range(len(sentence)):
            word=sentence[i]
            if (word not in self.word_dict):
                word = self.UNK
            temp_num = list(self.word_dict.keys()).index(word)
            if(i==0):
                score[i]=initial_matrix+launch_matrix.T[temp_num]
                path[i]=-1
            if(i>0):
                tp=launch_matrix.T[temp_num]
                temp_score=(tp+transfer_matrix).T+score[i-1]
                score[i]=np.max(temp_score,axis=1)
                path[i]=np.argmax(temp_score,axis=1)
        useful_tags=[]
        cur_num=len(sentence)-1
        step=int(np.argmax(score[cur_num]))
        useful_tags.insert(0, self.tag_count_dic[step])
        for k in range(cur_num, 0, -1):
            step = int(path[k][step])
            useful_tags.insert(0,self.tag_count_dic[step])
        return useful_tags

    def evaluate(self,dev_data):
        """
        评估函数，主要用于评价模型的好坏，输出开发集最终预测结果的准确率
        :return:
        """
        self.dev_set = data_set(dev_data)
        dev_sentences=self.dev_set.sentences
        all_correct_tags=self.dev_set.tags
        sentence_count=len(dev_sentences)
        words_all_count=0
        words_correct_count=0
        for i in range(sentence_count):
            sentence=dev_sentences[i]
            predict_tag=self.predict(sentence,self.alpha)
            correct_tag=all_correct_tags[i]
            words_all_count+=len(correct_tag)
            for j in range(len(correct_tag)):
                if(predict_tag[j]==correct_tag[j]):
                    words_correct_count+=1
        print("数据集%s一共有%d个句子。"%(dev_data,sentence_count))
        print("数据集%s一共有%d个单词，预测正确%d个单词。"%(dev_data,words_all_count,words_correct_count))
        print("数据集%s使用隐马尔可夫模型预测的准确率为:%.2f%%"%(dev_data,(words_correct_count/words_all_count)*100))








