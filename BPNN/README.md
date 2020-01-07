# Back Propagation Neural Network

## 结构

```sh
.
├── data
│   ├── ctb5
│   │   ├── dev.conll
│   │   ├── test.conll
│   │   └── train.conll
│   ├── ctb7
│   │   ├── dev.conll
│   │   ├── test.conll
│   │   └── train.conll
│   └── embed.txt
├── results
│   ├── abpnn.txt
│   └── bpnn.txt
├── bpnn.py
├── config.py
├── corpus.py
├── LICENSE
├── README.md
├── run.py
└── utils.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [--adagrad] [--threads THREADS] [--seed SEED] [--file FILE]

Create Back Propagation Neural Network(BPNN) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --adagrad, -a         use AdaGrad
  --threads THREADS, -t THREADS
                        set the max num of threads
  --seed SEED, -s SEED  set the seed for generating random numbers
  --file FILE, -f FILE  set where to store the model
# e.g. AdaGrad
$ python run.py --adagrad
```

## 结果

| AdaGrad | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :-----: | :------: | :------: | :------: | :------------: |
|    ×    |   7/18   | 93.8838% | 93.4259% | 0:05:19.534595 |
|    √    |  20/31   | 93.9474% | 93.3147% | 0:07:14.746511 |