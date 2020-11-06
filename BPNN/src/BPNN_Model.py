from date_set import dataset
from word_embeding import word_embeding
from active_function import *
import numpy as np
import datetime
from collections import defaultdict
import random
class BPNN_Model:
    def __init__(self,word_embeding_file,word_embeding_dim,windows,hidden_layer_size,train_file,dev_file,test_file=None):
        self.train_data=dataset(train_file)
        self.tag_num=self.train_data.tag_num#训练集中词性的数目
        self.tag_lst=self.train_data.tag #训练集中的词性列表
        self.tag_id=self.train_data.tag_id#训练集中词性-id的映射
        self.dev_data=dataset(dev_file)
        if (test_file != None):
            self.test_data = dataset(test_file)
        else:
            self.test_data = None
        self.embeding=word_embeding(word_embeding_file,word_embeding_dim)
        self.word_embed_dim = self.embeding.word_embed_dim#每个词向量的维度
        self.layer_size_lst=[windows*word_embeding_dim,hidden_layer_size,len(self.tag_lst)]#每一层神经元的数目
        self.layer_num=len(self.layer_size_lst)#神经网络的层数
        self.embeding.extend_word2vec(self.train_data)#利用train训练集来扩充词嵌入矩阵
        self.embeding_matrix=self.embeding.word_embed_matrix#词嵌入矩阵
        self.weight_lst=[np.random.randn(l2,l1)/np.sqrt(l1) for l1,l2 in zip(self.layer_size_lst[:-1],self.layer_size_lst[1:])]#随机初始化权重列表
        self.b_lst=[np.random.randn(l,1) for l in self.layer_size_lst[1:]]#随机初始化偏置项列表


    def forward_propagation(self,x,activation):
        ###前向传播
        z_lst=[]#经过wx+b，但还没有进入激活函数的每一层神经元的input列表
        a_lst=[]#经过激活函数的每一层神经元的output列表
        a=np.reshape(self.embeding_matrix[x],(-1,1))#生成一个（windows*word_embeding_dim,1）的输入矩阵，是由一个词以及前后的几个词的词向量拼接起来的
        a_lst.append(a)
        for weight,b in zip(self.weight_lst[:-1],self.b_lst[:-1]):
            temp_z=np.dot(weight,a)+b
            z_lst.append(temp_z)
            if(activation=="RELU"):
                a=RELU(temp_z)
            elif(activation=="sigmoid"):
                a=sigmoid(temp_z)
            a_lst.append(a)
        weight=self.weight_lst[-1]
        b=self.b_lst[-1]
        temp_z=np.dot(weight,a)+b
        z_lst.append(temp_z)
        a=softmax(temp_z)
        a_lst.append(a)
        return z_lst,a_lst



    def backward_propagation(self,a_lst,z_lst,y,activation):
        """
        反向传播过程（计算梯度）
        :param a_lst:经过wx+b，但还没有进入激活函数的每一层神经元的input列表
        :param z_lst:经过激活函数的每一层神经元的output列表
        :param y:正确的标签数据
        :param activation:激活函数
        :return:
        dw:每一个权重的梯度，是一个列表
        db:每一个偏置项的梯度，是一个列表
        dx:用于对训练集中词向量进行更新的梯度
        """
        db=[np.zeros(b.shape) for b in self.b_lst]
        dw=[np.zeros(weight.shape) for weight in self.weight_lst]
        dz=a_lst[-1]-y #交叉熵函数先对softmax层的a求导，a再对z求导，dz=a-y
        dw[-1]=np.dot(dz,a_lst[-2].T)#根据链式求导法则，dloss/dw[-1]=(dloss/dz[-1])*(dz[-1]/dw[-1]),而dz[-1]/dw[-1]=a[-2],要注意维度保持一致
        db[-1]=dz
        for i in range(2,self.layer_num):
            z=z_lst[-i]
            da=np.dot(self.weight_lst[-i+1].T,dz)
            if(activation=="RELU"):
                dz=da*back_RELU(z)
            elif(activation=="sigmoid"):
                dz=da*back_sigmoid(z)
            dw[-i]=np.dot(dz,a_lst[-i-1].T)
            db[-i]=dz
        dx=np.dot(self.weight_lst[0].T,dz)
        return dw,db,dx


    def compute_loss(self,data,lmbda,activation):
        """
        损失函数计算函数
        :param data:数据集
        :param lmbda:正则化系数
        :param activation:#激活函数
        :return:损失值
        """
        loss=0.0
        for x,y in data:
            z_lst,a_lst=self.forward_propagation(x,activation)#前向计算
            a=a_lst[-1]#softmax输出层的值，是一个向量
            loss-=np.log(a[np.argmax(y)])
        #在损失函数中计算正则项,这里采用了L2进行正则化表示
        loss+=0.5*lmbda*sum(np.linalg.norm(w)**2 for w in self.weight_lst)
        loss /= len(data)
        return loss

    def evaluate(self,data,activation):
        """
        模型评价函数
        :param data:数据集
        :param activation:激活函数
        :return: 返回总词数、预测正确的词数、正确率
        """
        total_num=len(data)
        predict_lst=[]
        for x,y in data:
            z_lst,a_lst=self.forward_propagation(x,activation)
            a=a_lst[-1]
            predict_lst.append(y[np.argmax(a)])
        correct_num=int(sum(predict_lst)[0])
        correct_percent=correct_num/total_num
        return correct_num,total_num,correct_percent


    def update(self,batch,activation,n,lmbda,learn_rate):
        """
        梯度下降更新权重
        :param batch:小规模样本数据
        :param activation:激活函数
        :param n:所有的样本数
        :param lmbda:正则项系数
        :param learn_rate:学习率
        :return:
        """
        batch_size=len(batch)#小规模样本中的数据数目
        dw=[np.zeros(w.shape) for w in self.weight_lst]#神经网络的权重梯度矩阵构成的列表
        db=[np.zeros(b.shape) for b in self.b_lst]#神经网络的偏置梯度矩阵构成的列表
        dx=defaultdict(float)#对于词向量的更新。采用dict的方式
        for x,y in batch:
            temp_z_lst,temp_a_lst=self.forward_propagation(x,activation)
            temp_dw,temp_db,temp_dx=self.backward_propagation(temp_a_lst,temp_z_lst,y,activation)
            dw=[dw[i]+temp_dw[i] for i in range(len(dw))]
            db=[db[i]+temp_db[i] for i in range(len(db))]
            for word_id,dg in zip(x,temp_dx.reshape(-1,self.word_embed_dim)):
                dx[word_id]+=dg
        self.weight_lst=[w-(w*learn_rate*(lmbda/n))-(learn_rate*grad/batch_size) for w,grad in zip(self.weight_lst,dw)]
        self.b_lst=[b-(learn_rate*grad/batch_size) for b,grad in zip(self.b_lst,db)]
        for id,g in dx.items():
            self.embeding_matrix[id]-=(learn_rate*g/batch_size)


    def SGD_training(self,iterations,max_iterations,learn_rate,lmbda,batch_size,decay,activation,windows,shuffle=False,regularization=False,Simulated_annealing=False):
        """
        SGD,随机梯度下降算法，每次更新小规模的数据
        :param iterations: 迭代次数
        :param max_iterations: 多少次迭代没有提升就退出
        :param learn_rate: 学习率
        :param lmbda: 正则化系数
        :param batch_size: 小规模样本大小
        :param decay: 学习率的衰退率
        :param activation: 激活函数
        :param windows: 窗口大小
        :param shuffle: 是否打乱数据
        :param regularization:  是否使用正则化
        :param Simulated_annealing: 是否使用模拟退火
        :return:
        """
        self.train = self.embeding.load_input_correct(windows, self.train_data,self.tag_num,self.tag_id) #将训练集数据转化为可以供神经网络进行输入的内容以及对应的正确的标签的矩阵。
        self.dev = self.embeding.load_input_correct(windows, self.dev_data,self.tag_num,self.tag_id)
        if(self.test_data!=None):
            self.test = self.embeding.load_input_correct(windows, self.test_data,self.tag_num,self.tag_id)
        counter=0
        max_step=100000#模拟退火的最大步数
        max_accuracy_rate=0.0#开发集的最大准确率
        max_accuracy_num=0#最高准确率对应第几次迭代
        step=0
        n=len(self.train)
        if(regularization):
            print("使用正则项对模型进行优化！")
        else:
            lmbda=0
        if(Simulated_annealing):
            print("使用模拟退火算法对模型进行优化！")
        else:
            decay=1
        for i in range(1,iterations+1):
            print("正在进行第%d轮迭代进行训练！"%(i))
            random.seed(6)
            starttime = datetime.datetime.now()
            if(shuffle==True):
                print("在这一轮迭代中打乱所有的训练集数据！")
                random.shuffle(self.train)
            batch_lst=[self.train[k:k+batch_size] for k in range(0,n,batch_size)]
            m=len(batch_lst)
            for batch in batch_lst:
                loss=self.update(batch,activation,m,lmbda,learn_rate* decay**(step/max_step))
                step+=1
            print("这一轮迭代已经完成，对于这一轮迭代的模型进行评估")
            #评价程序
            train_correct_num, total_num, train_precision = self.evaluate(self.train,activation)
            print('train(训练集)准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev,activation)
            print('dev(开发集)准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))
            if(self.test_data!=None):
                test_correct_num, test_num, test_precision = self.evaluate(self.test,activation)
                print('test(测试集)准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))
            if dev_precision > max_accuracy_rate:
                max_accuracy_rate = dev_precision
                max_accuracy_num=i
                counter=0
            else:
                counter+=1
            endtime = datetime.datetime.now()
            print("第%d次迭代所花费的时间为:%sS" % (i,endtime-starttime))
            if (counter>= max_iterations):
                break
        print('第%d次迭代对应的开发集预测的准确率最高，最高的准确率为:%f' % (max_accuracy_num, max_accuracy_rate))












