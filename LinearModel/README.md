## 线性模型Linear-Model

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 开发集
./big-data/
    train.conll: 训练集
    dev.conll: 开发集
    test.conll: 测试集
./result:
    origin.txt: 初始版本，小数据测试，使用W作为权重的评价结果
    origin-averaged.txt: 初始版本，小数据测试，使用V作为权重的评价结果
    partial.txt: 使用部分特征优化后，小数据测试，使用W作为权重的结果
    partial-averaged.txt: 使用部分特征优化后，小数据测试，使用V作为权重的结果
    bigdata-origin.txt: 初始版本，大数据测试，使用W作为权重的评价结果
    bigdata-origin-averaged.txt: 初始版本，大数据测试，使用V作为权重的评价结果
    bigdata-partial-feature.txt: 使用部分特征优化后，大数据测试，使用W作为权重的结果
    bigdata-partial-feature-averaged.txt: 使用部分特征优化后，大数据测试，使用V作为权重的结果
./src:
    linear-model.py: 初始版本的代码
    linear-model-partial-feature.py: 优化后的代码
    config.py: 配置文件，用字典存储每个参数
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```python
#配置文件中各个参数
config = {
    'train_data_file': './data/train.conll',   #训练集文件,大数据改为'./big-data/train.conll'
    'dev_data_file': './data/dev.conll',       #开发集文件,大数据改为'./big-data/dev.conll'
    'test_data_file': None,                    #测试集文件,大数据改为'./big-data/test.conll'
    'averaged': False,                         #是否使用averaged percetron
    'iterator': 20,                            #最大迭代次数
    'exitor': 10,                              #连续多少次迭代没有提升效果就退出
    'shuffle': False                           #每次迭代是否打乱数据
}
```

```bash
$ cd ./Linear-Model
$ python src/linear-model.py                   #修改config.py文件中的参数
$ python src/linear-model-partial-feature.py   #修改config.py文件中的参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 |  时间  |
| :-------------: | :----------------: | :------: | :------: | :-------: | :----: |
|        ×        |         ×          |    √     |  21/26   |  86.48%   | 17449s |
|        ×        |         √          |    √     |  22/22   |  87.68%   | 14477s |
|        √        |         ×          |    √     |  14/18   |  87.59%   |  789s  |
|        √        |         √          |    √     |  22/22   |  88.27%   | 1042s  |

注：代码参考了样例代码[Github链接](https://github.com/KiroSummer/LinearModel)。使用特征优化方法后，实际上拓宽了特征空间，所以准确率一般情况下都有所提高。由于Linear-Model本身过于简单，权重都是整数，在计算每个tag的score时很有可能出现许多个tag最大分值一样的情况，这时候取哪个作为预测的标签会影响最终的结果（误差在0.1%-0.2%，也可能是因为训练集太少的缘故)。在本例中，把所有的tag按字母序排列，从左到右编号，计算最大值时默认取第一个出现的最大值（np.argmax）。而在样例代码中，python2.7的词典顺序乱序，且取的是最后一个最大值。如果改为python3结果会出现较大的不一样！

##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | test准确率 |  时间  |
| :-------------: | :----------------: | :------: | :------: | :-------: | ---------- | :----: |
|        ×        |         ×          |    √     |  18/28   |  92.76%   | 92.26%     | 23218s |
|        ×        |         √          |    √     |  21/31   |  93.73%   | 93.57%     | 25041s |
|        √        |         ×          |    √     |  19/29   |  93.08%   | 92.66%     | 5678s  |
|        √        |         √          |    √     |  11/21   |  93.85%   | 93.68%     | 4108s  |

