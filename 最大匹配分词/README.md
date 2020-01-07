## 正向最大匹配分词

### 一.目录结构

```
./data/:
    data.conll: 数据文件
    data.txt: 生成的毛文本文件
    word_dict.txt: 生成的词典文件
    out.txt: 生成的分词结果文件
./src:
    CreateDataTxt.py: 创建毛文本
    CreateDataTxt.py: 创建词典文件
    WordSplit.py: 生成分词结果
    Evaluate.py: 评价结果
./README.md: 使用说明
```



### 二.运行

##### 1.运行环境

​    python 2.7

##### 2.运行方法

```bash
cd ./cd Max-Match-Word-Segmentation/
python src/CreateDataTxt.py    #创建毛文本
python src/CreateDataTxt.py    #创建词典文件
python src/WordSplit.py        #生成分词结果
python src/Evaluate.py         #评价结果
```

##### 3.参考结果

```
正确识别的词数：20263
识别出的总体个数：20397
测试集中的总体个数：20454
正确率：0.99343
召回率：0.99066
F值：0.99204
```



##### 