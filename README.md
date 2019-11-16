# TextNetwork
[![python](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org)  
[![jieba](https://img.shields.io/badge/jieba-0.39-yellowgreen)](https://github.com/fxsjy/jieba)
[![sklearn](https://img.shields.io/badge/sklearn-0.21.3-yellowgreen)](https://github.com/scikit-learn/scikit-learn)
[![networkx](https://img.shields.io/badge/networkx-2.4-yellowgreen)](https://github.com/networkx/networkx)  
通过网络(图)的视角对文本结构进行研究  
## 1.词共现网络
通过不同的度量方式与窗口长度,可以构造出不同结构的词共现网络.  
这里提供了两种不同的网络:  
1.1 以整个句子作为窗口,句子中的所有词均存在连接.
1.2 窗口长度为1,即只有两个相邻的词之间存在连接.
