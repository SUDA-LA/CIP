# Hidden Markov Model

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
│   ├── bhmm.txt
│   └── hmm.txt
├── config.py
├── hmm.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [--bigdata] [--file FILE]

Create Hidden Markov Model(HMM) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --bigdata, -b         use big data
  --file FILE, -f FILE  set where to store the model
# e.g. 
$ python run.py -b
```

## 结果

| 大数据集 | alpha |  dev/P   |  test/P  |     mT(s)      |
| :------: | :---: | :------: | :------: | :------------: |
|    ×     |  0.3  | 75.7428% |    *     | 0:00:00.723058 |
|    √     | 0.01  | 88.3546% | 88.4994% | 0:00:02.846564 |

