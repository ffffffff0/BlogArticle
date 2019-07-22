---
title: 决策树
date: 2018-2-20
tags: [Python, MachineLearn, DecisionTrees]
mathjax: true
---

决策树是一种基本的分类与回归的方法。分类决策树模型是一种描述对实例进行分类的树形结构。

<!-- more -->

### 信息增益

![decisiontree](http://image-1252432001.coscd.myqcloud.com/decisiontree.png)

上图中的决策什么时候终止，就涉及到一个最小的样本分割数量，何时停止划分数据集，如何划分数据集，这里就引入了信息增益。

划分数据集的大原则是：将无序的数据变得更加有序。在划分数据集之前之后信息发生的变化称为信息增益，计算每个特征值划分数据集获得的信息增益，其中信息增益最高的特征就是最好的选择。

#### 信息增益的计算

信息增益的计算，集合信息的度量方式称为香农熵或熵。

熵为信息的期望值，熵度量了事物的不确定性，越是不确定的事物，它的熵就越大。具体的，随机变量X的熵：
$$
H(X) = -\sum_{i=1}^n p_{i} log_{2}(p_{i})
$$
其中n代表X的n种不同的离散取值，而$p_{i}$ 代表了X取值为$i$ 的概率。

H(X) 度量了X的不确定性，H(X)的值越小，则X的纯度越高。

假设离散属性a有V个可能的取值${a^1, a^2,.....,a^v}$ ,若使用a来对样本集X进行划分，则会产生V个分支节点，其中第V个分支节点包含了X中所有在属性a上取值为$a^v$ 的样本，记为$X^v$ .可以计算出$X^v$ 的信息熵，由于不同的分支节点所包含的样本数不同，给分支节点赋予权重$\frac{|X^v|}{|X|}$ ,即样本数越多的分支节点的影响越大，于是可计算出属性a对样本X进行划分所获得"信息增益"。
$$
Gain(X, a) = H(X)  - \sum_{v=1}^V \frac{|X^v|}{|X|} H(X^v) 
$$

#### 信息增益计算举例

| 编号 | 色泽 | 根蒂 | 敲声 | 纹理 | 脐部 | 触感 | 好瓜 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | 青绿 | 蜷缩 | 浊响 | 清晰 | 凹陷 | 硬滑 | 是   |
| 2    | 乌黑 | 蜷缩 | 沉闷 | 清晰 | 凹陷 | 硬滑 | 是   |
| 3    | 乌黑 | 蜷缩 | 浊响 | 清晰 | 凹陷 | 硬滑 | 是   |
| 4    | 青绿 | 蜷缩 | 沉闷 | 清晰 | 凹陷 | 硬滑 | 是   |
| 5    | 浅白 | 蜷缩 | 浊响 | 清晰 | 凹陷 | 硬滑 | 是   |
| 6    | 青绿 | 稍蜷 | 浊响 | 清晰 | 稍凹 | 软粘 | 是   |
| 7    | 乌黑 | 稍蜷 | 浊响 | 稍糊 | 稍凹 | 软粘 | 是   |
| 8    | 乌黑 | 稍蜷 | 浊响 | 清晰 | 稍凹 | 硬滑 | 是   |
| 9    | 乌黑 | 稍蜷 | 沉闷 | 稍糊 | 稍凹 | 硬滑 | 否   |
| 10   | 青绿 | 硬挺 | 清脆 | 清晰 | 平坦 | 软粘 | 否   |
| 11   | 浅白 | 硬挺 | 清脆 | 模糊 | 平坦 | 硬滑 | 否   |
| 12   | 浅白 | 蜷缩 | 浊响 | 模糊 | 平坦 | 软粘 | 否   |
| 13   | 青绿 | 稍蜷 | 浊响 | 稍糊 | 凹陷 | 硬滑 | 否   |
| 14   | 浅白 | 稍蜷 | 沉闷 | 稍糊 | 凹陷 | 硬滑 | 否   |
| 15   | 乌黑 | 稍蜷 | 浊响 | 清晰 | 稍凹 | 软粘 | 否   |
| 16   | 浅白 | 蜷缩 | 浊响 | 模糊 | 平坦 | 硬滑 | 否   |
| 17   | 青绿 | 蜷缩 | 沉闷 | 稍糊 | 稍凹 | 硬滑 | 否   |

根据西瓜书上的例子可以熟悉和了解计算。

根据数据集易知n=2,根节点包含X中的所有样例，其中正例(即是好瓜)占$p_1 = \frac{8}{17}$ ,反例占$p_2 = \frac{9}{17}$ ,于是，可以计算出根节点的信息熵：


$$
H(X) = -\sum_{i=1}^2 p_i log_{2}(p_i) = -( \frac{8}{17}log_{2}(\frac{8}{17}) + \frac{9}{17} log_{2}(\frac{9}{17})) = 0.998
$$
从数据集中可以计算得到属性的集合{色泽，根蒂，敲声，纹理，脐部，触感}中每个属性的信息增益。以"纹理" 为例，它有3个可能的取值：{清晰，模糊，稍糊}，利用该属性对X进行划分，可得3个子集，分别记为：$X^1​$ (纹理=清晰)， $X^2​$ (纹理=稍糊)，$X^3​$ (纹理=模糊)。

$X^1$ 包含9个样例，其中正例占$p_1=\frac{7}{9}$ ,反例占$p_2=\frac{2}{9}$ ,$X^2$ 包含5个样例，其中正例占$p_1=\frac{1}{5}$ ,反例占$p_2=\frac{4}{5}$,$X^3$ 包含3个样例，其中正例占$p_1=\frac{0}{3}$ ,反例占$p_2=\frac{3}{3}$.

然后可以计算出3个分支节点的信息熵:


$$
H(X^1) = -(\frac{7}{9}log_{2}(\frac{7}{9}) + \frac{2}{9}log_{2}(\frac{2}{9})) = 0.764
$$

$$
H(X^2) = -(\frac{1}{5}log_{2}(\frac{1}{5}) + \frac{4}{5}log_{2}(\frac{4}{5})) = 0.721
$$

$$
H(X^3) = -(\frac{0}{3}log_{2}(\frac{0}{3}) + \frac{3}{3}log_{2}(\frac{3}{3})) = 0
$$

根据上面的推到可以得到属性”纹理“的信息增益为：
$$
Gain(X, 纹理) = H(X) - \sum_{i=1}^3 \frac{|X^i|}{|X|} H(X^i)
$$

$$
= 0.998 - (\frac{9}{17} \times 0.764 + \frac{5}{17} \times 0.721 + \frac{3}{17} \times 0) = 0.381
$$

类似的可以计算其他属性的信息增益：

$Gain(X, 根蒂) = 0.143$; $Gain(X, 敲声) = 0.141$；

$Gain(X, 纹理) = 0.381$; $Gain(X, 脐部) = 0.289$；

$Gain(X, 触感) = 0.006$；

显然，属性"纹理"的信息增益最大，于是它被选为划分属性的效果最好。

```python
                                    +---------+
                                    |  纹 理  |
                                    |         |
                                    +---+-----+
                                        |
                                        |
                   +--------------------+-------------------------+
                   | 清 晰            稍 糊                   模 糊|
                   |                    +                         |
                   |                    |                         |
                   |                    |                         |
                   |                    |                         |
+------------------+---+     +----------+----+    +---------------+
|{1,2,3,4,5,6,8,10,15} |     |{7,9,13,14,17} |    |{11,12,16}     |
+----------------------+     +---------------+    +---------------+

```

然后,利用决策树算法对每个节点作进一步划分，以上图中的("纹理=清晰")为例，该节点包含的样例集合$X^1$ 中有9个样例，可用属性集合为{色泽，根蒂，敲声，触感}，基于$X^1$ 计算出各个属性的信息增益:
$$
Gain(X^1, 色泽) = 0.043; Gain(X^1, 根蒂) = 0.458;
$$

$$
Gain(X^1, 敲声) = 0.331； Gain(X^1, 脐部) = 0.458；
$$

$$
Gain(X^1, 触感) = 0.458；
$$

其中"根蒂"， "脐部"， "触感", 3个属性均取得最大的信息增益，选其中一个来划分属性，同理，对每个节点进行上述操作，最终得到决策树如下：

```python
                                                   +-------------+
                                                   |  纹 理 = ?  |
                                                   +-----+-------+
                                                         |
                              +--------------------------+---------------------------------+
                              +                          稍 糊                               模 糊
                              清 晰                       +                                 +
                              |                          |                                 |
                     +--------+--------+         +-------+--------+                 +------+-----+
                     |  根 蒂 = ？      |         |   触 感 = ？   |                 |   坏 瓜     |
                     +--------+--------+         +-------+--------+                 +------------+
                              |                          |
                              |                          |
                              |                 +--------+-----------+
                              |                 硬 滑                 软 粘
                              |                 |                    |
                              |                 |                    |
                              |           +-----+---+          +-----+----+
                              |           | 好 瓜    |          | 坏 瓜    |
                              |           +---------+          +----------+
                              |
      +-----------------------+-----------------+
      蜷 缩                     稍 缩             硬 挺
      |                       |                 |
+-----+-----+          +------+-----+       +---+------+
|  好 瓜    |          | 色 泽 = ？  |       | 坏 瓜     |
+-----------+          +------+-----+       +----------+
                              |
                              |
          +-------------------+---------------+
          青 绿                 乌 黑             浅 白
          |                   |               |
    +-----+---+           +---+----+      +---+------+
    | 好 瓜    |          | 触 感 = ？|    | 好 瓜     |
    +---------+           +---+----+      +----------+
                              |
             +----------------+-----------------+
             硬 滑                                软 粘
             |                                  |
      +------+------+                 +---------+-----+
      |  好 瓜       |                 |  坏 瓜        |
      +-------------+                 +---------------+

```

### DecisionTrees 优缺点

决策树算法易于使用，结果更好的理解。但是决策树容易过度拟合，尤其使用包含大量特征的数据集时，适当的时间停止决策树的生长是很重要的。

### 代码实现

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
#### 划分数据集
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
#### 选择最好的特征
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
#### 构建决策树

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

#### 分类函数
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
