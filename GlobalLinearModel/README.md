# Global Linear Model

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
│   ├── aglm.txt
│   ├── baglm.txt
│   ├── bglm.txt
│   ├── boaglm.txt
│   ├── boglm.txt
│   ├── glm.txt
│   ├── oaglm.txt
│   └── oglm.txt
├── config.py
├── glm.py
├── oglm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [--bigdata] [--average] [--optimize] [--seed SEED]
              [--file FILE]

Create Global Linear Model(GLM) for POS Tagging.

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
|    ×     |      ×       |    ×     |  10/16   | 86.6989% |    *     | 0:00:22.911520 |
|    ×     |      ×       |    √     |  15/21   | 87.7025% |    *     | 0:00:22.918568 |
|    ×     |      √       |    ×     |  18/24   | 87.2970% |    *     | 0:00:05.806948 |
|    ×     |      √       |    √     |  18/24   | 88.0582% |    *     | 0:00:05.972707 |
|    √     |      ×       |    ×     |  21/32   | 93.4051% | 93.2555% | 0:14:48.366320 |
|    √     |      ×       |    √     |  16/27   | 94.2273% | 94.0217% | 0:14:14.846735 |
|    √     |      √       |    ×     |  10/21   | 93.6135% | 93.2371% | 0:03:04.171367 |
|    √     |      √       |    √     |  15/26   | 94.3157% | 94.0879% | 0:02:58.804926 |