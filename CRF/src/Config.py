# 配置文件

train_data_dir = "../data/train.conll"

dev_data_dir = "../data/dev.conll"

train_bigdata_dir = "../bigdata/train"

test_bigdata_dir = "../bigdata/test"

dev_bigdata_dir = "../bigdata/dev"

epoch = 50 # 最大迭代学习轮数

exitor = 5 # 迭代退出轮数

random_seed = 0 # 随机种子

shuffle = True # 是否打乱数据集

batch_size = 32 # 多少个样本更新一次权重

lmbda = 0.01 # L2正则化系数（不需要正则化则设置为0）

learning_rate = 0.3 # 学习率

decay_rate = 0.96 # 学习率衰减速率（用于模拟退火）（不需要模拟退火设置为1）



