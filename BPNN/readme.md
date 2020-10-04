# CRF

## 结构

```sh
.
├── data
│   ├── embed.txt
│   ├── train.conll
│   ├── dev.conll
│   └── test.conll
├── demo.py
├── Corpus.py
├── Tagger.py
├── config.py
├── config.ini
└── README.md
```

## 用法

```sh
# using vanilla tagger
$ python demo.py --data data -c config.ini
```

## 结果

|  iter          |  train  |  dev  |  test  |  time  |
| :------------: | :-----: | :---: | :----: | :----: |
<!-- TODO -->