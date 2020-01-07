# Log Linear Model

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
│   ├── allm.txt
│   ├── ballm.txt
│   ├── bllm.txt
│   ├── boallm.txt
│   ├── bollm.txt
│   ├── llm.txt
│   ├── oallm.txt
│   └── ollm.txt
├── config.py
├── llm.py
├── ollm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [--bigdata] [--anneal] [--optimize] [--regularize]
              [--seed SEED] [--file FILE]

Create Log Linear Model(LLM) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --bigdata, -b         use big data
  --anneal, -a          use simulated annealing
  --optimize, -o        use feature extracion optimization
  --regularize, -r      use L2 regularization
  --seed SEED, -s SEED  set the seed for generating random numbers
  --file FILE, -f FILE  set where to store the model
# e.g. 特征提取优化+模拟退火
$ python run.py -b --optimize --anneal
```

## 结果

| 大数据集 | 特征提取优化 | 模拟退火 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :------: | :----------: | :------: | :------: | :------: | :------: | :------------: |
|    ×     |      ×       |    ×     |  16/22   | 87.4898% |    *     | 0:00:19.078803 |
|    ×     |      ×       |    √     |  21/27   | 87.4918% |    *     | 0:00:20.170674 |
|    ×     |      √       |    ×     |  18/24   | 87.5256% |    *     | 0:00:04.068441 |
|    ×     |      √       |    √     |  12/18   | 87.5196% |    *     | 0:00:04.324282 |
|    √     |      ×       |    ×     |  24/35   | 93.8287% | 93.5473% | 0:14:29.164093 |
|    √     |      ×       |    √     |  27/38   | 93.8637% | 93.6159% | 0:13:58.444849 |
|    √     |      √       |    ×     |  23/34   | 93.8787% | 93.5632% | 0:02:43.619554 |
|    √     |      √       |    √     |  21/32   | 93.9721% | 93.6539% | 0:02:48.134954 |