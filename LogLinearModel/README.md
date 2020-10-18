## √对数线性模型Log_Linear_Model

### 一.目录结构

```
./src/:
   config.py #配置文件
   Log_Linear_Model.py #最开始写的对数线性模型，没有使用正则项、模拟退火和特征抽取优化，效率很低,梯度向量使用np.array
   Log_Linear_Model_improve.py #改进的对数线性模型,有正则项和模拟退火,无特征抽取优化,梯度向量用defaultdict,有提高效率
   Log_Linear_Model_partial.py #加入了特征抽取优化的对数线性模型，梯度向量使用np.array
   Log_Linear_Model_partial_improve.py #加入特征抽取优化对数线性模型,梯度向量使用defaultdict
   run.py #代码运行文件
./big_data/:
   train #大数据训练集
   dev #大数据开发集
   test #大数据测试集
./small_data/:
   train.conll #小数据训练集
   dev.conll #小数据开发集
./result/:
   original.txt #不使用特征抽取优化，模拟退火和正则项的结果
   original_reg_sim.txt #使用模拟退火和正则项，但不使用特征抽取优化的结果
   partial.txt #使用特征抽取优化，不使用模拟退火和正则项的结果
   partial_reg_sim.txt #使用特征抽取优化，模拟退火和正则项的结果
```

### 二.代码运行

#### 1.运行环境

python 3.8

#### 2.运行方法

```
#编辑config中参数
class config:
    def __init__(self,isbigdata):
        self.iterator=50 #最大迭代次数
        self.max_iterations=10 #迭代多少次没有提升就退出
        self.shuffle=True #是否打乱数据
        self.regularization=False #是否使用正则化进行优化
        self.Simulated_annealing=False #是否采用模拟退火
        self.eta=0.5 #初始学习率
        self.decay=0.96 #学习率的衰退系数
        self.C=0.0001 #正则化系数
        self.batch_size=50 #批次大小
        if(isbigdata):
            self.train="../big_data/train"
            self.test="../big_data/test"
            self.dev="../big_data/dev"
        else:
            self.train="../small_data/train.conll"
            self.dev="../small_data/dev.conll"
```

```
$ cd ./src
$ python3 run.py
```

#### 3.参考结果

##### （1）小数据集测试

| partial-feature | 初始学习率 | batch_size |  模拟退火  |  正则项  | 打乱数据 | 迭代次数 | dev准确率 | 时间 |
| :-------------: | :--------: | :--------: | :--------: | :------: | :------: | :------: | :-------: | :--: |
|        ×        |    0.5     |     40     |     ×      |    ×     |    √     |  27/38   |  87.46%   | 885  |
|        ×        |    0.5     |     40     | dacay=0.96 | C=0.0001 |    √     |  46/50   |  87.49%   | 1219 |
|        √        |    0.5     |     40     |     ×      |    ×     |    √     |   9/20   |  87.55%   |  67  |
|        √        |    0.5     |     40     | decay=0.96 | C=0.0001 |    √     |  24/35   |  87.60%   | 154  |

注：经过实验，发现对于Log_Linear_Model中的梯度向量g，如果我们采用np.array矩阵来存储的话，效率比较低，因为每次都要清空梯度向量，然后self.weight+=self.g，会有很多不必要的计算，但如果直接使用defaultdict，然后直接读取对应的关键字和值，加入对应的self.weight上时，效率会有很大的提升，迭代一次所需要的时间也大大减少。

##### （2）大数据集测试（还没有跑，计划后期在服务器上跑结果）

| partial-feature | 初始学习率 | batch_size | 退火 | 正则项 | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间 |
| :-------------: | :--------: | :--------: | :--: | :----: | :------: | :------: | :-------: | :--------: | :--: |
|        ×        |    0.5     |     40     |  ×   |   ×    |    √     |          |           |            |      |
|        ×        |    0.5     |     40     |  √   |   √    |    √     |          |           |            |      |
|        √        |    0.5     |     40     |  ×   |   ×    |    √     |          |           |            |      |
|        √        |    0.5     |     40     |  √   |   √    |    √     |          |           |            |      |

