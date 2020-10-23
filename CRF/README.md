## 条件随机场CRF（Condition-Random-Field）

### 一.目录结构

```
./src/:
   config.py #配置文件
   CRF.py #CRF模型,有正则项和模拟退火,无特征抽取优化,梯度向量用defaultdict
   CRF_partial.py #加入特征抽取优化的CRF模型,梯度向量使用defaultdict
   run.py #代码运行文件
./big_data/:
   train #大数据训练集
   dev #大数据开发集
   test #大数据测试集
./small_data/:
   train.conll #小数据训练集
   dev.conll #小数据开发集
./result/:
   CRF.txt #不使用特征抽取优化，模拟退火和正则项的结果
   CRF_reg_sim.txt #使用模拟退火和正则项，但不使用特征抽取优化的结果
   CRF_partial.txt #使用特征抽取优化，不使用模拟退火和正则项的结果
   CRF_partial_reg_sim.txt #使用特征抽取优化，模拟退火和正则项的结果
```

### 二.代码运行

#### 1.运行环境

python3.8

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
        self.C=0.00001 #正则化系数
        self.batch_size=1 #批次大小
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

| partial-feature | 初始学习率 | batch_size |  模拟退火  |  正则项   | 打乱数据 | 迭代次数 | dev准确率 | 时间 |
| :-------------: | :--------: | :--------: | :--------: | :-------: | :------: | :------: | :-------: | :--: |
|        ×        |    0.5     |     1      |     ×      |     ×     |    √     |  25/35   |  88.70%   | 1640 |
|        ×        |    0.5     |     1      | dacay=0.96 | C=0.00001 |    √     |  27/37   |  88.79%   | 1700 |
|        √        |    0.5     |     1      |     ×      |     ×     |    √     |   6/16   |  88.96%   | 322  |
|        √        |    0.5     |     1      | dacay=0.96 | C=0.00001 |    √     |  12/22   |  89.06%   | 465  |

##### （2）大数据集测试（还没有跑，计划后期在服务器上跑结果）

| partial-feature | 初始学习率 | batch_size | 退火 | 正则项 | 打乱数据 | 迭代次数 | dev准确率 | test准确率 |
| :-------------: | :--------: | :--------: | :--: | :----: | :------: | :------: | :-------: | :--------: |
|        ×        |    0.5     |     1      |  ×   |   ×    |    √     |          |           |            |
|        ×        |    0.5     |     1      |  √   |   √    |    √     |          |           |            |
|        √        |    0.5     |     1      |  ×   |   ×    |    √     |          |           |            |
|        √        |    0.5     |     1      |  √   |   √    |    √     |          |           |            |

