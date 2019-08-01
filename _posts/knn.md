---
title: KNN算法
date: 2018-2-21
tags: [Python, KNN, MachineLearn]
categories: 机器学习
---

KNN算法可用于分类问题和回归问题。然而，K – 最近邻算法更常用于分类问题。

#### KNN算法流程

在给定的数据集中，基于某种距离度量找出训练集中与其最靠近的K个训练样本，然后基于这k个“邻居”的信息来进行预测，一般情况下，在分类任务中可使用“投票法”，即选择这K个样本中出现最多的类别标记作为预测结果；在回归任务中可使用"平均法", 即将这K个样本的实值输出标记的平均值作为预测结果；还可以基于距离远近进行加权平均或加权投票，距离越近的样本权重越大。

<!-- more -->

距离度量有很多种，不同的度量方式，分类结果也会有显著不同。

西瓜书上有个图很能说明KNN的意义:

![knn](http://image-1252432001.coscd.myqcloud.com/knn.jpg)

#### SKLearn实现KNN

```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier
 
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model
modle = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
 
# Train the model using the training sets and check score
model.fit(X, y)
 
#Predict Output
pred = model.predict(x_test)
```

#### KNN优缺点

KNN算法原理非常简单，容易理解。但是KNN 的计算成本很高。变量应该先标准化（normalized），不然会被更高范围的变量偏倚。在使用KNN之前，要在异常值去除和噪音去除等前期处理多花功夫。