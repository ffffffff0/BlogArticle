---
title: 决策树Python的实现
date: 2019-7-9
tags: [Python, MachineLearn, DecisionTrees]
mathjax: true
---
> 机器学习实战的第三章的笔记；
<!--more-->
### 信息增益
度量方式为；**熵**

$$
H(X) = -\sum_{i=1}^n p_{i} log_{2}(p_{i})
$$
其中n代表X的n种不同的离散取值，而$p_{i}$ 代表了X取值为$i$ 的概率。
H(X) 度量了X的不确定性，H(X)的值越小，则X的纯度越高。

以如下数据集为例

| 1 | 1 | yes |
|---|---|-----|
| 1 | 1 | yes |
| 1 | 0 | no  |
| 0 | 1 | no  |
| 0 | 1 | no  |

**calcShannoEnt** 为计算一个数据集的熵的函数
```
# compute entropy 根节点的熵
def calcShannonEnt(dataSet):
    # 数据集的行数
    numEntries = len(dataSet)
    # 标签计数
    labelCounts = {}
    # 迭代每一行
    for featVec in dataSet:
        # 得到标签
        currentLabel = featVec[-1]
        # 判断标签是否存在
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        # labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    print(labelCounts)
    # 计算熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt
```
### 划分数据集
当得到熵时，可以按照最大的熵来划分数据集。在splitDataSet函数中axis为要匹配的那一行数据，返回的是值为value的除去axis这行以外的其他数据。

```
# splite dataset
def splitDataSet(dataSet, axis, value):
    '''
    dataSet 数据集
    axis 那一行的数据
    value 要提取的值
    '''
    retDataSet = []
    # 迭代数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将这一行的数据复制
            reducedFeatVec = featVec[:axis]
            print(reducedFeatVec)
            # 添加标签 
            print(featVec[axis+1:])
            reducedFeatVec.extend(featVec[axis + 1:])
            # 合并
            retDataSet.append(reducedFeatVec)
    return retDataSet
```
### 选择最好的特征
chooseBestFeatureToSplit 函数去选择最好的特征来分割数据集。
```
def chooseBestFeatureToSplit(dataSet):
    # all features
    numFeatures = len(dataSet[0]) - 1
    # dataset shannon entropy
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # feature list
        featList = [example[i] for example in dataSet]
        # feature value
        uniqueVals = set(featList)
        newEntorpy = 0.0
        # compute the i feature entropy
        for value in uniqueVals:
            # compute subdataset entropy
            subDataSet = splitDataSet(dataSet, i, value)
            # the feature entropy 
            prob = len(subDataSet) / float(len(dataSet))
            newEntorpy += prob * calcShannonEnt(subDataSet)
        # get the i feature entropy 
        infoGain = baseEntropy - newEntorpy
        # search best feature
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
```
### 构建决策树

递归的构建决策树，其中递归的结束条件为：
```
# final code to creat a tree
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 当数据集中只有一种属性时 
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 数据集已被划分完, 只有一个元素 
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 得到最佳特征和标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 构建树的类型
    myTree = {bestFeatLabel: {}}
    # 删除已用过的标签 
    del (labels[bestFeat])
    # 最佳特征的值 
    featValue = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        # 复制这个列表
        subLabels = labels[:]
        # 递归构建树 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTre
```
majorityCnt 函数返回classList 中出现次数最多的标签
```
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
        # if vote not in classCount.keys():
        #     classCount[vote] = 0
        # classCount[vote] + = 1
    sortedClassCount = sorted(classCount.items(), reverse=True)
    
    return sortedClassCount[0][0]
```

### 分类函数
```
def classify(inputTree, featLabels, testvec):
    '''
    inputTree 为生成的决策树
    featLabels 为特征标签
    testvect 为分类向量
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testvec)
            else:
                classLabel = secondDict[key]
    
    return classLabe
```
