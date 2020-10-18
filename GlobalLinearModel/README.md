## 全局线性模型Global_Linear_Model

### 一.目录结构

```
./src/:
   config.py #Global_Linear_Model的配置文件
   Global_Linear_Model.py #没有使用特征抽取优化的Global_Linear_Model
   Global_Linear_Model_improve.py #使用了特征抽取优化的Global_Linear_Model
   run.py #运行代码
./big_data/:
   train #大数据训练集
   dev #大数据开发集
   test #大数据测试集
./small_data/:
   train.conll #小数据训练集
   dev.conll #小数据开发集
./result/:
   original_global_weight.txt #小数据集，不采用partial feature和averaged percetron
   original_global_v.txt #小数据集，采用averaged percetron，不采用partical feature
   partical_global_weight.txt #小数据集，采用partical feature，不采用averaged percetron
   partical_global_v.txt #小数据集，采用partical feature和averaged percetron
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

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 |  时间  |
| :-------------: | :----------------: | :------: | :------: | :-------: | :----: |
|        ×        |         ×          |    √     |  16/26   |  87.00%   | 619(S) |
|        ×        |         √          |    √     |  28/38   |  87.73%   | 974(S) |
|        √        |         ×          |    √     |  14/24   |  87.43%   | 326(S) |
|        √        |         √          |    √     |  15/25   |  88.01%   | 431(S) |

注：Global_Linear_Model实际上是对于Linear_Model模型的一个优化和提升，Linear_Model模型只是利用了当前词的词性和词本身的特征，而Global_Linear_Model除了考虑当前词的词性以外，还要考虑它的前一个词的词性，即当前词的词性和前一个词的词性是相关的，从而预测出整个句子的词性序列，因此性能相比于Linear_Model有一个很大的提升。同时Global_Linear_Model模型还用到了维特比算法进行预测。Global_Linear_Model主要用于结构化分类问题。

##### （2）大数据集测试（还没有跑，计划后期在服务器上跑结果）

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间 |
| :-------------: | :----------------: | :------: | :------: | :-------: | :--------: | :--: |
|        ×        |         ×          |    √     |          |           |            |      |
|        ×        |         √          |    √     |          |           |            |      |
|        √        |         ×          |    √     |          |           |            |      |
|        √        |         √          |    √     |          |           |            |      |