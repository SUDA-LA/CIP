## 全局线性模型Global-Linear-Model

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
    bigdata-partial.txt: 使用部分特征优化后，大数据测试，使用W作为权重的结果
    bigdata-partial-averaged.txt: 使用部分特征优化后，大数据测试，使用V作为权重的结果
./src:
    global-linear-model.py: 初始版本的代码
    global-linear-model-partial-feature.py: 优化后的代码,速度还是过慢
    global-linear-model-partial-feature-2D.py: 进一步用numpy优化后的代码
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
    'test_data_file': './data/dev.conll',      #测试集文件,大数据改为'./big-data/test.conll'
    'averaged': False,                         #是否使用averaged percetron
    'iterator': 20,                            #最大迭代次数
    'exitor':10,                               #连续多少次迭代没有提升就退出
    'shuffle': False                           #每次迭代是否打乱数据
}
```

```bash
$ cd ./Global-Linear-Model
$ python src/global-linear-model.py                       #修改config.py文件中的参数
$ python src/global-linear-model-partial-feature-2D.py    #修改config.py文件中的参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :------: | :-------: | :-------: |
|        ×        |         ×          |    √     |  25/26   |  86.47%   |    40s    |
|        ×        |         √          |    √     |  18/21   |  87.65%   |    40s    |
|        √        |         ×          |    √     |  14/14   |  87.37%   |    18s    |
|        √        |         √          |    √     |  14/16   |  87.99%   |    18s    |



##### (2)大数据测试

训练集：big-data/train.conll,共46,572个句子,共1,057,943个词。

开发集：big-data/dev.conll,共2,079个句子,共59,955个词。

测试集：big-data/test.conll,共2,796个句子,共81,578个词。

| partial feature | averaged percetron | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :------: | :-------: | ---------- | :-------: |
|        √        |         ×          |    √     |  17/27   |  93.62%   | 93.28%     |   ~7min   |
|        √        |         √          |    √     |  11/21   |  94.40%   | 94.16%     |   ~7min   |

