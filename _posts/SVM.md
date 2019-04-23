---
title: 支持向量机
date: 2018-2-19
tags: [SVM, Python, MachineLearn]
mathjax: true
---

支持向量机属于监督学习算法，除了进行线性分类之外，SVM还可以使用核函数有效地进行非线性分类，将其输入隐式映射到高维特征空间中。

<!-- more -->

####  SKLearn中的SVM

首先是分类：

```python
from sklearn import svm
# 训练特征
X = [[0, 0], [1, 1]]
# 训练标签
y = [0, 1]
# 创建一个分类器
clf = svm.SVC(kernel="linear")
# 使用训练特征和训练标签进行拟合
clf.fit(X, y) 
# output: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
# decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
# max_iter=-1, probability=False, random_state=None, shrinking=True,
# tol=0.001, verbose=False)
# 预测
clf.predict([[2., 2.]])
# output: array([1])
```

#### 非线性SVM

```python
# 对于线性SVM输入特征，返回label
          +------------+
+----X---^+            |
          |    SVM     +----Label
+----Y---^+            |
          +------------+
```

![svm](http://image-1252432001.coscd.myqcloud.com/svm.png)

对于上图的情况，如何分类就牵扯出非线性分类，我们可以改变输入特征，从上面的X和Y改为
$$
X^2+Y^2
$$

```python
                 +--------------+
    Z=X^2+Y^2    |              |
+---------------^+     SVM      +-------Label--->
                 |              |
                 +--------------+
```

由此可以得到一个新的特征Z=X^2+Y^2,然后特征可以转换为下图：

![svm2](http://image-1252432001.coscd.myqcloud.com/svm2.png)

上图是显然非常线性可分的，上图的分割函数是一条直线，对于原来的特征来说应该是一条类似圆的曲线。

#### 核技巧(kernel trick)

支持向量机中有一种叫做核技巧的东西，核技巧就是获取低维度输入空间或者特征空间，并将其映射到极高维度空间的函数，将线性不可分转化为线性可分，这种函数称为核函数。

SKLearn 中的核函数 默认为rbf型，其他可选的有linear，poly，sigmoid，precomputed，以及可调用自定义形式callable。如果给出了一个可调用函数，则用于从数据矩阵预先计算核心矩阵; 该矩阵应该是一个数组其形式为（n_samples，n_samples）。

#### SVM的C参数

C参数是SVM中非常重要的一个参数，它控制决策边界的光滑和正确分类所有训练点之间的折衷。

C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

#### 过度拟合

![svm3](http://image-1252432001.coscd.myqcloud.com/svm3.png)

上图就是一个过度拟合的例子，其实上图分明是线性可分的，但是过度的拟合会形成上图的结果，其中C参数和Kernel设置的不正确会显著的导致过度拟合。

#### SVM的优缺点

支持向量机在具有复杂领域和明显分隔边界的条件下表现十分出色，但是在海量数据的情况下，SVM表现并不够好，数据噪声过多也同样，在这种情况下，朴素贝叶斯表现很好。

有时可以减少训练样本，来寻找合适的C参数和Kernel，来提高准确率和训练速度。