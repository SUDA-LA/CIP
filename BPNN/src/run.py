from BPNN_Model import BPNN_Model
from config import config
import datetime
if __name__ == '__main__':
    st = input("请输入你要训练和测试的数据集，b表示大数据集，c表示小数据集\n")
    if (st == "b"):
        isbigdata = True
    else:
        isbigdata = False
    confi=config(isbigdata)
    iterations=confi.iterations#最大迭代次数
    max_iterations=confi.max_iterations#迭代多少次没有提升就退出
    learn_rate=confi.learn_rate#初始学习率
    lmbda=confi.lmbda#正则化系数
    batch_size=confi.batch_size#更新批次大小
    decay=confi.decay#学习率的衰退系数
    activation=confi.activation#激活函数
    windows=confi.windows#窗口大小
    shuffle=confi.shuffle#是否打乱数据
    regularization=confi.regularization#是否使用正则化进行优化
    Simulated_annealing=confi.Simulated_annealing#是否采用模拟退火
    word_embeding_file=confi.word_embeding_file#预训练的词向量文件
    word_embeding_dim=confi.word_embeding_dim#词向量维度
    hidden_layer_size=confi.hidden_layer_size#隐藏层维度
    ftrain=confi.train
    fdev=confi.dev
    if (isbigdata):
        print("使用大数据集进行训练和预测!")
        ftest=confi.test#测试集
        starttime = datetime.datetime.now()
        bpnn=BPNN_Model(word_embeding_file,word_embeding_dim,windows,hidden_layer_size,ftrain,fdev,ftest)
        bpnn.SGD_training(iterations,max_iterations,learn_rate,lmbda,batch_size,decay,activation,windows,shuffle,regularization,Simulated_annealing)
        endtime = datetime.datetime.now()
        print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime - starttime).seconds)))
    else:
        print("使用小数据集进行训练和预测!")
        starttime = datetime.datetime.now()
        bpnn=BPNN_Model(word_embeding_file,word_embeding_dim,windows,hidden_layer_size,ftrain,fdev)
        bpnn.SGD_training(iterations,max_iterations,learn_rate,lmbda,batch_size,decay,activation,windows,shuffle,regularization,Simulated_annealing)
        endtime = datetime.datetime.now()
        print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime - starttime).seconds)))
