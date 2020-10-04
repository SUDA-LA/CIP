# Hidden Markov Model

## 结构

```sh
.
├── data
│   ├── train.conll
│   └── dev.conll
├── bigdata
│   ├── train.conll
│   ├── dev.conll
│   └── test.conll
├── demo.py
├── DataReader.py
├── Tagger.py
├── config.py
├── bigdata_config.ini
├── data_config.ini
└── README.md
```

## 用法

```sh
# training on data
$ python demo.py -c data_config.ini

Set the seed for built-in generating random numbers to 1
Set the seed for numpy generating random numbers to 1
train accuracy: 92.27535%
dev   accuracy: 75.74077%
spend time: 0:00:02.495958s

Tagging Accuracy: 0.75741

# training on bigdata
$ python demo.py --data bigdata  -c bigdata_config.ini

Set the seed for built-in generating random numbers to 1
Set the seed for numpy generating random numbers to 1
train accuracy: 93.70703%
dev   accuracy: 88.35460%
test  accuracy: 88.49935%
spend time: 0:00:50.414834s

Tagging Accuracy: 0.88499
```

## 结果

| 大数据集 | alpha |  dev/ACC   |  test/ACC  |     mT(s)      |
| :------: | :---: | :------: | :------: | :------------: |
|    ×     |  0.3  | 75.741% |    --   | 0:00:02.495958 |
|    √     | 0.01  | 88.354% | 88.499% | 0:00:50.414834 |