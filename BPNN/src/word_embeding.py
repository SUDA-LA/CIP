import numpy as np
########这个类（程序）主要是用于对词向量和词嵌入矩阵的处理
class word_embeding:
    def __init__(self,filename,word_embed_dim):
        """
        初始化函数
        :param filename: 存放已经预训练过的的词向量
        :param word_embed_dim: 词向量的维度
        """
        self.nhit='<NHIT>'#未登录词
        self.start='<START>'#
        self.end='<END>'
        self.nhit_id=0
        self.start_id=0
        self.end_id=0
        self.mark=['<NHIT>','<START>','<END>']

        self.word_embed_dim=word_embed_dim
        self.word_num=0
        self.id_word={}
        self.word_id={}
        self.word_embed_matrix=self.load_word2vec(filename)


    def load_word2vec(self,filename):
        """
        将预训练好的词向量文件读取，然后转化为一个二维矩阵，词语的编号与词语的词向量相对应
        再加入三个标记词，分别为NHIT,START,END，表示未命中的词，开头的词，结束的词
        :param filename: 预训练好的词向量文件
        :return: 词嵌入矩阵
        """
        f=open(filename, "rb")
        count=0
        word2vec_matrix=[]
        for line in f:#读取已经训练好的词向量文件，转化为词向量矩阵
            temp=line.decode('utf-8').rstrip('\n').split()
            word2vec_matrix.append(list(map(float,temp[1:])))
            self.id_word[count]=temp[0]
            self.word_id[temp[0]]=count
            count+=1
        f.close()
        #在词向量矩阵中添加三种特殊情况（开头，结尾，未登录词）
        for mark in self.mark:
            if mark not in self.word_id.keys():
                self.word_id[mark] = count
                self.id_word[count] = mark
                word2vec_matrix.append((np.random.randn(self.word_embed_dim) / np.sqrt(self.word_embed_dim)).tolist())
                count+=1
        word2vec_matrix=np.array(word2vec_matrix)
        self.word_num=len(self.word_id)
        self.nhit_id=self.word_id[self.nhit]
        self.start_id=self.word_id[self.start]
        self.end_id=self.word_id[self.end]
        print("已成功完成对于词向量矩阵的初始化，词向量矩阵中有%d个词，每个词向量的维度为%d"%(self.word_num,self.word_embed_dim))
        return word2vec_matrix

    def extend_word2vec(self,use_data):
        """
        利用我们的数据集，对于词嵌入矩阵进行扩充，在数据集中，但不在词嵌入矩阵中的词，通过随机生成100维的词向量，进行扩充，之后在训练神经网络时跟着一起更新
        :param use_data: 数据集
        :return:
        """
        nhit_word=[word for word in use_data.word if word not in self.word_id.keys()]
        nhit_word_num=len(nhit_word)
        for id in range(nhit_word_num):
            id_now=id+self.word_num
            self.word_id[nhit_word[id]]=id_now
            self.id_word[id_now]=nhit_word[id]
        self.word_embed_matrix=np.concatenate([self.word_embed_matrix,np.random.randn(nhit_word_num, self.word_embed_dim)/np.sqrt(self.word_embed_dim)], axis=0)#数据集中的未登录词加入到词向量矩阵
        self.word_num=len(self.word_id)
        print("通过对数据集%s进行搜索，我们为当前词向量矩阵增加了%d个未登录词，当前词向量矩阵中有%d个词"%(use_data.filename,nhit_word_num,self.word_num))

    def load_input_correct(self,windows,use_data,tag_num,tag2id):
        """
        将我们的数据集转化为我们所需要的输入层和与输出层一一对应的正确值。
        :param windows: 窗口大小
        :param use_data: 数据集
        :param tag_num: 输出层维度
        :param tag2id: 词性和id的映射
        :return: 经过处理可以直接使用的数据集合
        """
        data=[]
        half=windows//2
        all_sentences=use_data.sentences
        for sentence in all_sentences:
            word_id_lst=[self.start_id]*half+[self.word_id.get(kk[0],self.nhit_id) for kk in sentence]+[self.end_id]*half
            tag_id_lst=[tag2id[kk[1]] for kk in sentence]
            for k in range(len(tag_id_lst)):
                temp_x=word_id_lst[k:k+windows]
                temp_y=self.one_hot(tag_id_lst[k],tag_num)
                data.append((temp_x,temp_y))
        print("数据集%s载入完成，成功转化为神经网络中可以使用的输入输出层信息，数据集样本规模大小为%d"%(use_data.filename,len(data)))
        return data

    def one_hot(self,id,tag_num):
        """
        对于输出进行one-hot编码，把一个词性标签编码成tag_num维的one-hot矩阵
        :param id:词性在词性列表中的索引，位置
        :param tag_num:输出层的维度
        :return:一个one-hot矩阵
        """
        vetor=np.zeros((tag_num,1))
        vetor[id]=1.0
        return vetor



