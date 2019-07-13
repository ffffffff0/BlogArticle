---
title: 朴素贝叶斯
date: 2018-2-17
tags: [Bayes, Python, MachineLearn]
mathjax: true
---

朴素贝叶斯算法是监督学习算法的一种，属于一种线性分类器，理解朴素贝叶斯就要先说贝叶斯定理。

## 贝叶斯定理

**条件概率**：常常要计算在某个事件B发生的条件下，另一个事件A发生的概率。概率论中称此概率为事件B已发生的条件下事件A发生的条件概率，记为P(A|B)。
<!-- more -->
$P(A|B) = \frac{P(A\cap B)}{P(B)}$
同理可得：
$P(A\cap B) = P(A|B)P(B)$

$P(A\cap B) = P(B|A)P(A)$

即:
$P(A|B)P(B) = P(B|A)P(A)$

**全概率公式**：假定样本空间S，是两个事件A与A'的和。



![img](http://www.ruanyifeng.com/blogimg/asset/201108/bg2011082503.jpg)

图中，红色部分是事件A，绿色部分是事件A'，它们共同构成了样本空间S。

在这种情况下，事件B可以划分成两个部分。

![img](http://www.ruanyifeng.com/blogimg/asset/201108/bg2011082504.jpg)

可以写出B的概率：
$P(B) = P(B\cap A) + P(B\cap A')$
由上条件概率可得：
$P(B) = P(B|A)P(A) + P(B|A')P(A')$
这就是全概公式。

到这里贝叶斯定理也就呼之欲出了,


$P(A|B) =  \frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|A')P(A')}$

## 朴素贝叶斯分类器

朴素贝叶斯分类器是一种基于贝叶斯定理的弱分类器，所有朴素贝叶斯分类器都假定样本每个特征与其他特征都不相关。

也就是为什么叫 Naive Bayes.

举个例子，如果一种水果其具有红，圆，直径大概3英寸等特征，该水果可以被判定为是苹果。尽管这些特征相互依赖或者有些特征由其他特征决定，然而朴素贝叶斯分类器认为这些属性在判定该水果是否为苹果的概率分布上独立的。

条件概率可以变形得到：
$P(A|B) = \frac {P(A\cap B)}{P(B)} = P(A)\frac {P(B|A)}{P(B)}$
其中P(A)称为"先验概率"，即在B事件发生之前，我们对A事件概率的一个判断。P(A|B)称为"后验概率"，即在B事件发生之后，我们对A事件概率的重新评估。P(B|A)/P(B)称为"可能性函数"，这是一个调整因子，使得预估概率更接近真实概率。

```python
后验概率　＝　先验概率 ｘ 调整因子
```

## 朴素贝叶斯分类案例
> 使用朴素贝叶斯进行文档分类

利用python构建一个分类器，对某条语句进行判断，它是否属于侮辱类的语句，分别用1和0表示。

首先要构建一个单词向量，对句子转为向量，考虑句子中出现的单词来判断语句的类型。

### 构建词向量
利用已经标注好的文本来训练模型。

loadDataSet() 用于加载数据集
```
def loadDataSet():
    """
    函数功能: 
        创建数据集
    参数: 
         无
    返回:
        postingList: 划分好的数据
        clasive: 数据的标签
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    # 类标签向量 1 代表侮辱性词汇 0 非侮辱性词汇
    clasive = [0, 1, 0, 1, 0, 1]
    return postingList, clasiv
```
createVocabList 函数创建一个单词不重复的列表。
```
def createVocabList(dataSet):
    """
    函数功能:
        创建一个不重复词的列表
    参数:
        dataSet: 分割好的词汇
    返回:
        不重复词的列表
    """
    vocabSet = set([])
    for document in dataSet:
        # 两个集合的交集
        vocabSet = vocabSet | set(document)
    
    return list(vocabSet)
```
setOfWordsVec 函数利用单词列表将 inpuSet 向量化
```
def setOfWords2Vec(vocabList, inputSet):
    """
    函数功能:
        根据vocabList将inputSet向量化，向量的每个元素为1或0
    参数:
        vocabList: 不重复的词列表
        inputSet: 切分好的词
    返回:
        文档向量
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocavulary!' % word)

    return returnVec
```
### 构建训练函数
利用构建好的单词向量来训练函数。
```
def trainNB0(trainMatrix, trainCategory):
    """
    函数功能:
        朴素贝叶斯分类器训练函数
    参数:
        trainMatrix: 训练文档矩阵
        trainCategory: 训练类别的标签向量
    返回:
        p0Vect: 非侮辱类的条件概率数组
        p1Vect: 侮辱类的条件概率数组
        pAbusive: 文档属于侮辱类的概率
    """
    # 训练文档的数量
    numTrainDocs = len(trainMatrix)
    # 每条文档的词条数
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历每一个文档
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect 单词在侮辱类文档中出现的概率
    # p0Vect 单词在非侮辱类文档中出现的概率
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive
```
getTrainMat 函数获得训练的向量列表
```
def getTrainMat(dataSet):
    """
    函数功能:
        生成训练集的向量列表
    参数:
        dataSet: 分割好的词列表
    返回:
        trainMat: 所有词条向量组成的列表
    """
    trainMat = []
    vocabList = createVocabList(dataSet)
    for inputSet in dataSet:
        returnVec = setOfWords2Vec(vocabList, inputSet)
        trainMat.append(returnVec)
    
    return trainMa
```
运行如下程序可得
```
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
myTrainMat = getTrainMat(listOPosts)
p0V, p1V, pAb = trainNB0(myTrainMat, listClasses)
```
![one](https://image-1252432001.cos.ap-chengdu.myqcloud.com/bayes/one.bmp)

### 测试分类器
classifyNB 函数为贝叶斯分类函数
```
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    函数功能:
        朴素贝叶斯分类器分类函数
    参数:
        vec2Classify: 待分类的词条数组 
        p0Vec: 非侮辱类的条件概率数组
        p1Vec: 侮辱类的条件概率数组
        pClass1: 文档属于侮辱类的概率
    返回:
        0: 非侮辱类
        1: 侮辱类
    """
    # 没有计算分母
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0- pClass1)
    # 判读概率大小
    if p1 > p0:
        return 1
    else:
        return 0
```
构建测试函数
$$
p(侮辱类|['stupid', 'garbage']) = \frac{p(侮辱类)p(['stupid', 'garbade']|侮辱类)}{p(['stupid', 'garbage'])}
$$

其中只需要比较 $p(侮辱类|['stupid', 'garbage'])$ 与 $p(非侮辱类|['stupid', 'garbage'])$ 的大小，故不必求解 $p(['stupid', 'garbage']$
```
def testingNB():
    """
    函数功能:
        朴素贝叶斯测试函数
    参数:
        无
    返回:
        无
    """
    # 创建实验样本
    dataSet, classVec = loadDataSet()
    # 创建词汇表
    myVocabList = createVocabList(dataSet)
    # 实验样本向量化
    trainMat = getTrainMat(dataSet)
    # 训练分类器
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # 测试样例
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
```

### 分类器的改进
利用贝叶斯分类器进行分类时，计算多个概率的乘积来得到文档属于某个类别的概率， 即$p(w_0| 侮辱类), p(w_1| 侮辱类)$。 如果其中一个概率值为0，那么最后也为0，这显然是不对的，为了改变可以将所有词数的初始为1，分母初始为2，这种做法叫做拉普拉斯平滑。（代码中已更改)

另外就是，很多小数相乘时，程序会下溢或者得到错误的结果，为了解决可以对乘积结果取自然对数，通过求对数可以避免下溢或者精度的错误。而且采用对数不会有任何的损失。

![two](https://image-1252432001.cos.ap-chengdu.myqcloud.com/bayes/two.bmp)

观察上图可以发现在相同的区域中的增减性一致。