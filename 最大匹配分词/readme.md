# Max Matching

## 结构

```sh
.
├── data
│   └── data.conll
├── demo.py
├── process_conll.py
├── Segmenter.py
├── WordDictionary.py
├── ValUtils.py
└── README.md
```

## 文件功能

* `demo.py`: 主函数
* `process_conll.py`: 数据预处理
* `Segmenter.py`: 分词器类
* `WordDictionary.py`: 词典类
* `ValUtils.py`: 分词评价函数

## 用法

```sh
# preprocess the data.conll
# from conll to:
#   raw sentence in ./data/data.txt
#   gold segmentation in ./data/ground_truth.json
#   word dictonary in ./data/word.dict
$ python process_conll.py

$ python demo.py

precision: 0.994 recall: 0.991 F-measure: 0.992
```