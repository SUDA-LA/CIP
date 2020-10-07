##  章岳—二元隐马尔科夫模型

### 目录结构
```
├─src
|  ├─Config.py：配置文件
|  ├─data_loader.py：数据读取类代码
|  ├─hmm_model.py：HMM模型代码
|  ├─run.py：运行文件
├─data
|  ├─dev.conll
|  └train.conll
├─bigdata
|    ├─dev
|    ├─test
|    └train
```

### 运行方式

#### 环境

Python3.8

#### 命令

```shell
cd ./src
python3 run.py
```

### 运行结果

| 数据集             | alpha | 总词数 | 标注正确的词数 | 标注准确率 | 耗时        |
| ------------------ | ----- | ------ | -------------- | ---------- | ----------- |
| 测试集（小训练集） | 0.3   | 50319  | 38110          | 0.757368   | 2.015657(s) |
| 开发集（大训练集） | 0.01  | 20454  | 18139          | 0.886819   | 4.332345(s) |
| 测试集（大训练集） | 0.01  | 50319  | 44355          | 0.881476   | 5.220471(s) |

### 学习过程中的笔记和思考

[NLP自学笔记：隐马尔可夫模型](https://hillzhang1999.gitee.io/2020/03/17/nlp-zi-xue-bi-ji-yin-ma-er-ke-fu-mo-xing/)

### 参考资料（部分）

- [李正华老师的课件](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/main.pdf)
- 李航老师的《统计学习方法》第10章
- [HMM学习最佳范例](http://www.comp.leeds.ac.uk/roger/HiddenMarkovModels/html_dev/main.html)
- [隐马尔科夫模型（Hidden Markov Model，HMM）](https://blog.csdn.net/lukabruce/article/details/82380511)
- [一站式解决：隐马尔可夫模型（HMM）全过程推导及实现](https://zhuanlan.zhihu.com/p/85454896)
- [如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？](https://www.zhihu.com/question/35866596)
- [如何用简单易懂的例子解释隐马尔可夫模型？](https://www.zhihu.com/question/20962240)

