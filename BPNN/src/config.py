class config:
    def __init__(self, isbigdata):
        self.iterations=60 #最大迭代次数
        self.max_iterations=20 #迭代多少次没有提升就退出
        self.learn_rate=0.5#初始学习率
        self.lmbda=0.01#正则化系数
        self.batch_size=50#更新批次大小
        self.decay=0.96#学习率的衰退系数
        self.activation="RELU"#激活函数
        self.windows=5#窗口大小
        self.shuffle=True #是否打乱数据
        self.regularization=True #是否使用正则化进行优化
        self.Simulated_annealing=True #是否采用模拟退火
        self.word_embeding_file="../pretrain/giga.100.txt"#预训练的词向量文件
        self.word_embeding_dim=100#词向量维度
        self.hidden_layer_size=300#隐藏层维度
        if (isbigdata):
            self.train="../big_data/train"
            self.test="../big_data/test"
            self.dev="../big_data/dev"
        else:
            self.train="../small_data/train.conll"
            self.dev="../small_data/dev.conll"