# Linear Model

## 结构

```sh
.
├── bigdata
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── data
│   ├── dev.conll
│   └── train.conll
├── results
│   ├── alm.txt
│   ├── balm.txt
│   ├── blm.txt
│   ├── boalm.txt
│   ├── bolm.txt
│   ├── lm.txt
│   ├── oalm.txt
│   └── olm.txt
├── config.py
├── lm.py
├── olm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [--bigdata] [--average] [--optimize] [--seed SEED]
              [--file FILE]

Create Linear Model(LM) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --bigdata, -b         use big data
  --average, -a         use average perceptron
  --optimize, -o        use feature extracion optimization
  --seed SEED, -s SEED  set the seed for generating random numbers
  --file FILE, -f FILE  set where to store the model
# e.g. 特征提取优化+权重累加
$ python run.py -b --optimize --average
```

## 结果

| 大数据集 | 特征提取优化 | 权重累加 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :------: | :----------: | :------: | :------: | :------: | :------: | :------------: |
|    ×     |      ×       |    ×     |  19/25   | 85.1567% |    *     | 0:00:17.302251 |
|    ×     |      ×       |    √     |  10/16   | 85.8443% |    *     | 0:00:17.485643 |
|    ×     |      √       |    ×     |  12/18   | 85.7449% |    *     | 0:00:02.615718 |
|    ×     |      √       |    √     |  12/18   | 86.1027% |    *     | 0:00:02.444157 |
|    √     |      ×       |    ×     |  46/57   | 92.9430% | 92.6708% | 0:10:37.877909 |
|    √     |      ×       |    √     |  22/33   | 93.7970% | 93.4933% | 0:10:42.888698 |
|    √     |      √       |    ×     |   9/20   | 92.9280% | 92.5960% | 0:01:09.899912 |
|    √     |      √       |    √     |  17/28   | 93.8420% | 93.6919% | 0:01:09.888973 |