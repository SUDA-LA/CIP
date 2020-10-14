## 线性模型Linear_Model

### 一.目录结构：

```
./src/:
   config.py #Linear_Model配置文件
   Linear_Model.py #没有使用特征抽取优化的Linear_Model
   Linear_Model_improved.py #使用了特征抽取优化的Linear_Model
   run.py #运行代码
./big_data/:
   train #大数据训练集
   dev #大数据开发集
   test #大数据测试集
./small_data/:
   train.conll #小数据训练集
   dev.conll #小数据开发集
./result/:
   original_weight.txt #小数据集，不采用partial feature和averaged percetron
   original_v.txt #小数据集，采用averaged percetron，不采用partical feature
   partical_weight.txt #小数据集，采用partical feature，不采用averaged percetron
   partical_v.txt #小数据集，采用partical feature和averaged percetron
```

### 二.代码运行

#### 1.运行环境：

python 3.8

#### 2.运行方法：

```
#编辑config中参数
class config:
    def __init__(self,isbigdata):
        self.shuttle=True #是否打乱数据
        self.average_perceptron=True #是否使用self.v
        self.iterations=50 #设置最大迭代次数
        self.max_iterations=10 #设置迭代多少次没有提升就退出
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

#### 3.参考结果：

##### （1）小数据集测试

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | 时间 |
| :-------------: | :----------------: | :------: | :------: | :-------: | :--: |
|        ×        |         ×          |    √     |  12/13   |  85.16%   | 260S |
|        ×        |         √          |    √     |  14/16   |  85.81%   | 316S |
|        √        |         ×          |    √     |   9/14   |  85.72%   | 25S  |
|        √        |         √          |    √     |  11/11   |  86.02%   | 54S  |

注：在使用shuffle对数据集合进行打乱时发现，每一次运行，最终得到的最好模型的dev准确率都不一样，有0.5%左右的浮动。猜测可能是因为我们线性模型是使用online_training的方法进行训练，每次对于权重的更新都是以1为单位的，因此可能会出现有多个相同分数的词性，这时候我们只能取到第一个词性，因此可能出现预测的错误。因为数据在不断地打乱，所以可能有时候运行的时候，能够取到正确结果，准确率高，有时候不能取到正确结果，准确率下降。

##### （2）大数据集测试（还没有跑，计划后期在服务器上跑结果）

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间 |
| :-------------: | :----------------: | :------: | :------: | :-------: | :--------: | :--: |
|        ×        |         ×          |    √     |          |           |            |      |
|        ×        |         √          |    √     |          |           |            |      |
|        √        |         ×          |    √     |          |           |            |      |
|        √        |         √          |    √     |          |           |            |      |

