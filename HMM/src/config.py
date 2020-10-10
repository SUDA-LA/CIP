class config:
    def __init__(self,isbigdata):
        if(isbigdata):
            self.alpha=0.01#平滑参数
            self.train="big_data/train"
            self.test="big_data/test"
            self.dev="big_data/dev"
        else:
            self.alpha=0.3
            self.train="small_data/train.conll"
            self.dev="small_data/dev.conll"