# Max Matching

## 结构

```sh
.
├── data
│   └── data.conll
├── maxmatch.py
└── README.md
```

## 用法

```sh
$ python maxmatch.py
Create dict of given data
A total of 4537 different words, of which the max len is 10
Create text of given data
Segment the words in text
Evaluate the result
Precision: 20263 / 20397 = 0.993430
Recall: 20263 / 20454 = 0.990662
F-value: 0.993430 * 0.990662 * 2 / (0.993430 + 0.990662) = 0.992044
```