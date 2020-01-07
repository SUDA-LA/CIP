## 一阶隐马尔可夫模型HMM

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 测试集
./big-data/:
    train.conll: 大数据训练集
    dev.conll: 大数据开发集
    test.conll: 大数据测试集
./src:
    config.py: 配置文件
    HMM.py: 一阶隐马尔可夫模型的代码
./predict.txt: 预测结果
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```
#配置文件中各个参数
config = {
    'train_data_file': './data/train.conll',   #训练集文件,大数据改为'./big-data/train.conll'
    'test_data_file': './data/dev.conll',     #测试集文件,大数据改为'./big-data/test.conll'
                                               #或者'./big-data/dev.conll'
    'predict_file': './predict_txt',           #模型预测结果文件
    'alpha': 0.3                               #平滑参数
}
```

```bash
$ cd ./HMM
$ python src/HMM.py			#修改config.py中的参数
```

##### 3.参考结果

##### (1)小数据测试

注：可以修改不同的alpha比较准确率。

| 训练集    | ./data/train.conll |
| --------- | ------------------ |
| 测试集    | ./data/dev.conll   |
| 参数alpha | 0.3                |
| 准确率    | 75.74%             |
| 执行时间  | 3s              |

##### (2)大数据测试

注：可以修改不同的alpha比较准确率

| 训练集    | ./big-data/train.conll | ./big-data/train.conll |
| --------- | ---------------------- | ---------------------- |
| 测试集    | ./big-data/dev.conll   | ./big-data/test.conll  |
| 参数alpha | 0.01                   | 0.01                   |
| 准确率    | 88.35%                 | 88.50%                 |
| 执行时间  | 40s                  | 58s                  |



