from HMM import HMM
from config import config
import datetime
if __name__ == '__main__':
    st=input("请输入你要训练和测试的数据集，b表示大数据集，c表示小数据集\n")
    if(st=="b"):
        isbigdata=True
    else:
        isbigdata=False
    confi=config(isbigdata)
    ftrain=confi.train
    fdev=confi.dev
    alpha=confi.alpha
    if(isbigdata):
        print("使用大数据集进行训练和预测，alpha=%f"%(alpha))
        ftest=confi.test
        hmm=HMM(ftrain,alpha)
        hmm.achieve_train_data_set()
        hmm.training(hmm.alpha)
        starttime = datetime.datetime.now()
        hmm.evaluate(ftest)
        endtime = datetime.datetime.now()
        print("预测test集共花费时间"+str((endtime-starttime).seconds)+"S")
        starttime = datetime.datetime.now()
        hmm.evaluate(fdev)
        endtime = datetime.datetime.now()
        print("预测dev集共花费时间" + str((endtime - starttime).seconds) + "S")
    else:
        print("使用小数据集进行训练和预测，alpha=%f"%(alpha))
        hmm = HMM(ftrain, alpha)
        hmm.achieve_train_data_set()
        hmm.training(hmm.alpha)
        starttime = datetime.datetime.now()
        hmm.evaluate(fdev)
        endtime = datetime.datetime.now()
        print("预测dev集共花费时间" + str((endtime - starttime).seconds) + "S")





