from Global_Linear_Model import Global_Linear_Model
from Global_Linear_Model_improve import Global_Linear_Model_improve
from config import config
import datetime
if __name__ == '__main__':
    st=input("请输入你要训练和测试的数据集，b表示大数据集，c表示小数据集\n")
    if(st=="b"):
        isbigdata=True
    else:
        isbigdata=False
    confi=config(isbigdata)
    iterations=confi.iterations
    max_iterations=confi.max_iterations
    shuttle=confi.shuttle
    average_perceptron=confi.average_perceptron
    ftrain=confi.train
    fdev=confi.dev
    ys=input("是否使用特征抽取优化，y代表是，n代表不是\n")
    if(isbigdata):
        print("使用大数据集进行训练和预测!")
        ftest=confi.test
        if(ys=="y"):
            print("使用特征抽取优化进行预测!")
            starttime=datetime.datetime.now()
            global_linear_model=Global_Linear_Model_improve(ftrain,fdev,ftest)
            global_linear_model.create_feature_space()
            global_linear_model.Online_Training(iterations,max_iterations,average_perceptron,shuttle)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒"%(fdev.split("/")[-1],str((endtime-starttime).seconds)))
        else:
            print("不使用特征抽取优化进行预测!")
            starttime = datetime.datetime.now()
            global_linear_model=Global_Linear_Model(ftrain,fdev,ftest)
            global_linear_model.create_feature_space()
            global_linear_model.Online_Training(iterations,max_iterations,average_perceptron,shuttle)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒" % (fdev.split("/")[-1], str((endtime-starttime).seconds)))
    else:
        print("使用小数据集进行训练和预测!")
        if(ys=="y"):
            print("使用特征抽取优化进行预测!")
            starttime=datetime.datetime.now()
            global_linear_model=Global_Linear_Model_improve(ftrain,fdev)
            global_linear_model.create_feature_space()
            global_linear_model.Online_Training(iterations,max_iterations,average_perceptron,shuttle)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒"%(fdev.split("/")[-1],str((endtime-starttime).seconds)))
        else:
            print("不使用特征抽取优化进行预测!")
            starttime = datetime.datetime.now()
            global_linear_model=Global_Linear_Model(ftrain,fdev)
            global_linear_model.create_feature_space()
            global_linear_model.Online_Training(iterations,max_iterations,average_perceptron,shuttle)
            endtime = datetime.datetime.now()
            print("本次训练和预测%s所花费的总时间是：%s秒"%(fdev.split("/")[-1],str((endtime-starttime).seconds)))