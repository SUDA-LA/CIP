class config:
    def __init__(self,isbigdata):
        if(isbigdata):
            self.alpha=0.01#平滑参数
            self.train="train"
            self.test="test"
            self.dev="dev"
        else:
            self.alpha=0.3
            self.train="train.conll"
            self.dev="dev.conll"