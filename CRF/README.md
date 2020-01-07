## 条件随机场Condition-Random-Field

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
    origin.txt: 初始版本，小数据测试
    partial.txt: 使用部分特征优化后，小数据测试
./src:
    CRF.py: 初始版本的代码
    CRF-partial-feature.py: 使用特征优化的代码
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
    'train_data_file': './data/train.conll',   # 训练集文件,大数据改为'./big-data/train.conll'
    'dev_data_file': './data/dev.conll',       # 开发集文件,大数据改为'./big-data/dev.conll'
    'test_data_file': './data/dev.conll',      # 测试集文件,大数据改为'./big-data/test.conll'
    'batchsize': 1,                            # 是否使用averaged percetron
    'iterator': 100,                           # 最大迭代次数
    'shuffle': False                           # 每次迭代是否打乱数据
    'regulization': False,                     # 是否正则化
    'step_opt': False,                         # 是否步长优化
    'exitor': 10,                              # 连续多少个迭代没有提升就退出
    'eta': 0.5,                                # 初始步长
    'C': 0.0001                                # 正则化系数,regulization为False时无效
}
```

```bash
$ cd ./CRF
$ python src/CRF.py                    #修改config.py文件中的参数
$ python src/CRF-partial-feature.py    #修改config.py文件中的参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| partial-feature | 初始步长 | batch-size | 步长优化 |  正则化   | 打乱数据 | 迭代次数 | dev准确率 | 时间/迭代 |
| :-------------: | -------- | :--------: | :------: | :-------: | :------: | :------: | :-------: | :-------: |
|        ×        | 0.5      |     1      |    ×     |     ×     |    √     |  26/36   |  88.68%   |    82s    |
|        ×        | 0.5      |     1      |    √     | C=0.00001 |    √     |  26/36   |  88.70%   |    70s    |
|        √        | 0.5      |     1      |    ×     |     ×     |    √     |   9/19   |  88.94%   |    17s    |
|        √        | 0.5      |     1      |    √     | C=0.00001 |    √     |  10/20   |  88.96%   |    17s    |



##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| partial-feature | 初始步长 | batch-size | 步长优化 | 正则化 | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间/迭代 |
| :-------------: | -------- | :--------: | :------: | :----: | :------: | :------: | :-------: | ---------- | :-------: |
|        √        | 0.2      |     1      |    ×     |   ×    |    √     |  11/21   |  94.28%   | 93.92%     |  ~13min   |
|        √        | 0.1      |     30     |    ×     |   ×    |    √     |  12/22   |  94.38%   | 94.18%     |  ~13min   |



​	