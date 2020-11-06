## 反向传播神经网络模型BPNN

### 一.目录结构

```
./src/:
   config.py #配置文件
   active_function.py #激活函数代码文件
   date_set.py #数据处理文件
   word_embeding.py #词向量、词嵌入矩阵处理文件
   BPNN_Model.py #反向传播（BPNN）模型
   run.py #代码运行文件
./big_data/:
   train #大数据训练集
   dev #大数据开发集
   test #大数据测试集
./small_data/:
   train.conll #小数据训练集
   dev.conll #小数据开发集
./result/:
   sig_reg_bpnn.txt #加入了正则化，模拟退火的结果
./pretrain/:
   giga.100.txt #预训练的词向量文件
```

### 二.代码运行

#### 1.运行环境

python 3.8

#### 2.运行方法

```
#编辑config中参数
class config:
    def __init__(self, isbigdata):
        self.iterations=60 #最大迭代次数
        self.max_iterations=20 #迭代多少次没有提升就退出
        self.learn_rate=0.5#初始学习率
        self.lmbda=0.01#正则化系数
        self.batch_size=50#更新批次大小
        self.decay=0.96#学习率的衰退系数
        self.activation="RELU"#激活函数
        self.windows=5#窗口大小
        self.shuffle=True #是否打乱数据
        self.regularization=True #是否使用正则化进行优化
        self.Simulated_annealing=True #是否采用模拟退火
        self.word_embeding_file="../pretrain/giga.100.txt"#预训练的词向量文件
        self.word_embeding_dim=100#词向量维度
        self.hidden_layer_size=300#隐藏层维度
        if (isbigdata):
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

#### 3.参数设置

|         参数名称         |       变量名        |    参数值    |
| :----------------------: | :-----------------: | :----------: |
|         迭代次数         |     iterations      |      60      |
| 迭代多少次没有提升就退出 |   max_iterations    |      20      |
|        初始学习率        |     learn_rate      |     0.5      |
|        正则项系数        |        lmbda        |     0.01     |
|       更新批次大小       |     batch_size      |      50      |
|     学习率的衰退系数     |        decay        |     0.96     |
|         激活函数         |     activation      |     RELU     |
|         窗口大小         |       windows       |      5       |
|       是否打乱数据       |       shuffle       |     True     |
|      是否使用正则化      |   regularization    |     True     |
|     是否采用模拟退火     | Simulated_annealing |     True     |
|    预训练的词向量文件    | word_embeding_file  | giga.100.txt |
|        词向量维度        |  word_embeding_dim  |     100      |
|        隐藏层维度        |  hidden_layer_size  |     300      |
|       大数据训练集       |      big_train      |    train     |
|       大数据开发集       |       big_dev       |     dev      |
|       大数据测试集       |      big_test       |     test     |
|       小数据训练集       |     small_train     | train.conll  |
|       小数据测试集       |     small_test      |  dev.conll   |

#### 4.实验结果

##### （1）小数据集测试

| 初始学习率 | batch_size | 模拟退火 | 正则项 | 打乱数据 | 迭代次数 | dev准确率 | 时间 |
| :--------: | :--------: | :------: | :----: | :------: | :------: | :-------: | :--: |
|    0.5     |     50     |    √     |   √    |    √     |  33/53   |  89.27%   | 1208 |

##### （2）大数据集测试（还没有跑，计划后期在服务器上跑结果）

| 初始学习率 | batch_size | 模拟退火 | 正则项 | 打乱数据 | 迭代次数 | dev准确率 | test准确率 | 时间 |
| :--------: | :--------: | :------: | :----: | :------: | :------: | :-------: | :--------: | :--: |
|    0.5     |     50     |    √     |   √    |    √     |          |           |            |      |