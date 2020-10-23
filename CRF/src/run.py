from CRF import CRF
from CRF_partial import CRF_partial
from config import config
import datetime

if __name__ == '__main__':
    st=input("请输入你要训练和测试的数据集，b表示大数据集，c表示小数据集\n")
    if(st=="b"):
        isbigdata=True
    else:
        isbigdata=False
    confi=config(isbigdata)
    iterator=confi.iterator #最大迭代次数
    max_iterations=confi.max_iterations #迭代多少次没有提升就退出
    decay=confi.decay #衰减系数
    Simulated_annealing=confi.Simulated_annealing #是否使用模拟退火
    regularization=confi.regularization #是否使用正则化
    batch_size=confi.batch_size #批次大小（多久更新一次self.weight）
    shuffle=confi.shuffle #是否打乱数据顺序
    eta=confi.eta #学习率
    C=confi.C #正则项系数
    ftrain=confi.train #训练集
    fdev=confi.dev #开发集
    ys=input("是否使用特征抽取优化，y代表是，n代表不是\n")
    if (isbigdata):
        print("使用大数据集进行训练和预测!")
        ftest = confi.test #测试集
        if (ys=="y"):
            print("使用特征抽取优化进行预测!")
            starttime = datetime.datetime.now()
            crf=CRF_partial(ftrain,fdev,ftest)
            crf.create_feature_space()
            crf.SGD_Training(iterator,max_iterations,eta,batch_size,C,decay,shuffle,regularization,Simulated_annealing)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime-starttime).seconds)))
        else:
            print("不使用特征抽取优化进行预测!")
            starttime = datetime.datetime.now()
            crf=CRF(ftrain, fdev, ftest)
            crf.create_feature_space()
            crf.SGD_Training(iterator, max_iterations, eta, batch_size, C, decay, shuffle, regularization,Simulated_annealing)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime - starttime).seconds)))
    else:
        print("使用小数据集进行训练和预测!")
        if (ys=="y"):
            print("使用特征抽取优化进行预测!")
            starttime=datetime.datetime.now()
            crf=CRF_partial(ftrain,fdev)
            crf.create_feature_space()
            crf.SGD_Training(iterator,max_iterations,eta,batch_size,C,decay,shuffle,regularization,Simulated_annealing)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime-starttime).seconds)))
        else:
            print("不使用特征抽取优化进行预测!")
            starttime=datetime.datetime.now()
            crf=CRF(ftrain, fdev)
            crf.create_feature_space()
            crf.SGD_Training(iterator, max_iterations, eta, batch_size, C, decay, shuffle, regularization,Simulated_annealing)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime - starttime).seconds)))



