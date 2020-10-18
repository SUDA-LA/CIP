class config:
    def __init__(self,isbigdata):
        self.shuttle=True#是否需要打乱数据
        self.average_perceptron=False#是否使用self.v
        self.iterations=50#最大迭代次数
        self.max_iterations=10#迭代多少次没有提升就退出
        if(isbigdata):
            self.train="../big_data/train"
            self.test="../big_data/test"
            self.dev="../big_data/dev"
        else:
            self.train="../small_data/train.conll"
            self.dev="../small_data/dev.conll"