# 中文信息处理基础练习

Teacher: 张民、李正华

Teach Assistant: 张月、巢佳媛、陈伟、凡子威

2015秋季学期，计算机学院大四本科生，选修课

上课时间：每周四9:00-12:00；地点：东教楼310

上机课时间：单周周二10:10-12:00；地点：理工楼239

## 用法

```sh
# 复制仓库到本地并进入目录
$ git clone git@github.com:SUDA-LA/CIP.git && cd CIP
# 切换到相应的分支
$ git checkout -b <branch> origin/<branch>
# ...
# 主分支有变动请注意及时更新(optional)
# $ git merge origin/master
# ...
# 提交代码到自己的分支(不属于你的分支无法提交)
$ git push
```

## 练习列表

* 汉字编码（C/C++语言实现）
* 最大匹配分词
* 网页正文抽取，请见2016春季学期《信息检索》课程主页
* HMM ([slides](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/main.pdf))
* LinearModel ([slides](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/9-linear-model/main2.pdf))
* LogLinearModel ([slides](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/10-maxent-loglinear/main.pdf))
* GlobalLinearModel ([slides](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/11-global-linear-model/main.pdf))
* CRF ([slides](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/12-crf/main.pdf))

## 可选练习

* BPNN ([slides](https://github.com/SUDA-LA/CIP/blob/master/BPNN/slides/Deep_Learning_for_POSTagging.pptx))
  * 推荐阅读（[英文版](http://neuralnetworksanddeeplearning.com/index.html)；[中文版](https://github.com/zhanggyb/nndl/releases/download/latest/nndl-ebook.pdf)）
  * PyTorch相关学习材料（[Deep Learning for NLP with Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)）
* 序列标注
  * 基于BiLSTM+CRF的序列标注模型（POS Tagging，NER和chunking），在此基础上利用了CharLSTM和[ELMo](https://allennlp.org/elmo)，可以参考[蒋炜](https://github.com/HMJW/Sequence-Labelling)和[张宇](https://github.com/zysite/tagger)同学的仓库
