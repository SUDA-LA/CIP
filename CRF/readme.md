# CRF

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
├── OptimizedTagger.py
├── config.py
├── config.ini
├── optimized_config.ini
└── README.md
```

## 用法

```sh
# using vanilla tagger
$ python demo.py --data data -c config.ini

# using optimized tagger
$ python demo.py --optimized --data data -c optimized_config.ini
```

## 结果

| 大数据集  | 特征优化 | 模拟退火  | 随机学习率 |  iter          |  train  |  dev  |  test  |  time  |
| :------: | :---:  | :------: | :------: | :------------: | :-----: | :---: | :----: | :----: |
<!-- TODO -->