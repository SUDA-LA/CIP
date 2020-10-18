class config:
    def __init__(self,isbigdata):
        self.iterator=50 #最大迭代次数
        self.max_iterations=10 #迭代多少次没有提升就退出
        self.shuffle=True #是否打乱数据
        self.regularization=True #是否使用正则化进行优化
        self.Simulated_annealing=True #是否采用模拟退火
        self.eta=0.5 #初始学习率
        self.decay=0.96 #学习率的衰退系数
        self.C=0.0001 #正则化系数
        self.batch_size=40 #批次大小
        if(isbigdata):
            self.train="../big_data/train"
            self.test="../big_data/test"
            self.dev="../big_data/dev"
        else:
            self.train="../small_data/train.conll"
            self.dev="../small_data/dev.conll"