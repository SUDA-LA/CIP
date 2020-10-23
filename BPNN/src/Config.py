# 配置文件

train_data_dir = "../data/train.conll"

dev_data_dir = "../data/dev.conll"

train_bigdata_dir = "../bigdata/train"

test_bigdata_dir = "../bigdata/test"

dev_bigdata_dir = "../bigdata/dev"

word2vec_dir = "../corpus/giga.100.txt"

char2vec_dir = "../corpus/giga.chars.100.txt"

epoch = 50 # 最大迭代学习轮数

exitor = 10 # 迭代退出轮数

random_seed = 0 # 随机种子

shuffle = True # 是否打乱数据集

embedding_freeze = False # 是否冻结词向量层

activation = 'ReLU' # 隐藏层激活函数选择

batch_size = 32 # 多少个样本更新一次权重

window = 5 # 上下文窗口大小

lmbda = 0.01 # L2正则化系数（不需要正则化则设置为0）

learning_rate = 0.5 # 学习率

decay_rate = 0.96 # 学习率衰减速率（用于模拟退火）（不需要模拟退火设置为1）

word_embed_dim = 100 # 词向量维度

char_embed_dim = 100 # 字向量维度

layer_sizes = [word_embed_dim * window, 300] # 神经元维度

mode = 's' # 数据集选择
