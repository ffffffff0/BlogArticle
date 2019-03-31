---
title: 随机森林
date: 2018-02-21
tags: [Python, MachineLearn, RandomForests]
---

随机森林是表示决策树总体的一个专有名词。在随机森林算法中，有一系列的决策树。为了根据一个新对象的属性将其分类，每一个决策树有一个分类，称之为这个决策树“投票”给该分类。这个森林选择获得森林里（在所有树中）获得票数最多的分类。

<!-- more -->

#### Bagging

Bagging(Bootstrap aggregating)方法是集成学习(ensemble methods)中获得用于训练基分类器(base estimator)的数据的重要一环，Bagging方法放进一个黑色的`bag`中，黑色意味着我们看不到里面的数据的详细情况，只知道里面有我们的数据集。然后从这个`bag`中随机抽一部分数据出来用于训练一个基分类器(base estimator)。抽到的数据用完之后我们有两种选择，放回或不放回。

既然样本本身可以`bagging`，那么特征是不是也可以`bagging`呢？当然可以！`bagging`完数据本身之后我们可以再`bagging` 特征，即从所有特征维度里面随机选取部分特征用于训练。在后面我们会看到，这两个**‘随机’**就是随机森林的精髓所在。从随机性来看，`bagging`技术可以有效的减小方差，即减小过拟合程度。

在SKLearn中，我们可以很方便的将`bagging`技术应用于一个分类器，提高性能：

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(n_estimators=50, bootstrap=True, KNeighborsClassifier(), bootstrap_features=True, max_samples=0.5, max_samples=0.5)
```

#### 随机森林的生成

bagging + decision trees = randomfroest, 随机森林中，将决策树作为基分类器(base estimator),  然后采用bagging技术训练一大堆小决策树， 最后将这些小决策树组合起来，这样就得到了森林。

随机森林中有许多的分类树。我们要将一个输入样本进行分类，我们需要将输入样本输入到每棵树中进行分类。打个形象的比喻：森林中召开会议，讨论某个动物到底是老鼠还是松鼠，每棵树都要独立地发表自己对这个问题的看法，也就是每棵树都要投票。该动物到底是老鼠还是松鼠，要依据投票情况来确定，获得票数最多的类别就是森林的分类结果。森林中的每棵树都是独立的，99.9%不相关的树做出的预测结果涵盖所有的情况，这些预测结果将会彼此抵消。少数优秀的树的预测结果将会超脱于芸芸“噪音”，做出一个好的预测。将若干个弱分类器的分类结果进行投票选择，从而组成一个强分类器，这就是随机森林bagging的思想（关于bagging的一个有必要提及的问题：bagging的代价是不用单棵决策树来做预测，具体哪个变量起到重要作用变得未知，所以bagging改进了预测准确率但损失了解释性。）

有了树我们就可以分类了，但是森林中的每棵树是怎么生成的呢？

每棵树的按照如下规则生成：

1）如果训练集大小为N，对于每棵树而言，随机且有放回地从训练集中的抽取N个训练样本（这种采样方式称为bootstrap sample方法），作为该树的训练集；

从这里我们可以知道：每棵树的训练集都是不同的，而且里面包含重复的训练样本。

**为什么要随机抽样训练集？**

如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的，这样的话完全没有bagging的必要；

**为什么要有放回地抽样?**

如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是"有偏的"，都是绝对"片面的"（当然这样说可能不对），也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决，这种表决应该是"求同"，因此使用完全不同的训练集来训练每棵树这样对最终分类结果是没有帮助的，这样无异于是"盲人摸象"。

2）如果每个样本的特征维度为M，指定一个常数m<<M，随机地从M个特征中选取m个特征子集，每次树进行分裂时，从这m个特征中选择最优的；

3）每棵树都尽最大程度的生长，并且没有剪枝过程。

随机森林中的“随机”就是指的这里的两个随机性。两个随机性的引入对随机森林的分类性能至关重要。由于它们的引入，使得随机森林不容易陷入过拟合，并且具有很好得抗噪能力（比如：对缺省值不敏感）。

#### 袋外错误率（oob error）

随机森林的分类效果的影响因素有：

> 森林中任意两棵树的相关性：相关性越大，错误率越大
>
> 森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。

减小特征选择个数m，树的相关性和分类能力也会相应的降低；增大m，两者也会随之增大。所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。

构建随机森林的关键问题就是如何选择最优的m，要解决这个问题主要依据计算袋外错误率oob error(out-of-bag error)

在构建每棵树时，我们对训练集使用了不同的bootstrap sample（随机且有放回地抽取）。所以对于每棵树而言（假设对于第k棵树），大约有1/3的训练实例没有参与第k棵树的生成，它们称为第k棵树的oob样本.

为什么要引入这个概念？因为在实际中数据通常是异常宝贵的，按照传统流程我们要将从数据集中分出一部分作为验证集，进而用验证集来调参。在随机森林中既然每棵树都有`OOB`样本.用`OOB error`代替验证集错误，在实践中效果非常好，更大的一点好处是节省了验证集数据开销.

#### 随机森林举例

**描述**：根据已有的训练集已经生成了对应的随机森林，随机森林如何利用某一个人的年龄（Age）、性别（Gender）、教育情况（Highest Educational Qualification）、工作领域（Industry）以及住宅地（Residence）共5个字段来预测他的收入层次。

**收入层次 :**

　　　　Band 1 : Below $40,000

　　　　Band 2: $40,000 – 150,000

　　　　Band 3: More than $150,000

　　随机森林中每一棵树都可以看做是一棵分类回归树，这里假设森林中有5棵树，总特征个数N=5，我们取m=1（这里假设每个树对应一个不同的特征）。

**CART 1 : Variable Age**

![rf1](http://image-1252432001.coscd.myqcloud.com/RandomForests/rf1.png)

**CART 2 : Variable Gender**

　　![rf2](http://image-1252432001.coscd.myqcloud.com/RandomForests/rf2.png)

**CART 3 : Variable Education**

　　![rf3](http://image-1252432001.coscd.myqcloud.com/RandomForests/rf3.png)

**CART 4 : Variable Residence**

　　![rf4](http://image-1252432001.coscd.myqcloud.com/RandomForests/rf4.png)

**CART 5 : Variable Industry**

　　![rf5](http://image-1252432001.coscd.myqcloud.com/RandomForests/rf5.png)



我们要预测的某个人的信息如下：

　　1. Age : 35 years ; 2. Gender : Male ; 3. Highest Educational Qualification : Diploma holder; 4. Industry : Manufacturing; 5. Residence : Metro.

　　根据这五棵树的分类结果，我们可以针对这个人的信息建立收入层次的分布情况：

　　![DF](http://image-1252432001.coscd.myqcloud.com/RandomForests/DF.png)

　　最后，我们得出结论，这个人的收入层次70%是一等，大约24%为二等，6%为三等，所以最终认定该人属于一等收入层次（小于$40,000）。

#### SKLearn 随机森林

```python
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
```



### 参考文章

[**随机森林(Random Forest)**](http://www.cnblogs.com/maybe2030/p/4585705.html)

[**机器学习算法之随机森林（Random Forest）**](http://backnode.github.io/pages/2015/04/23/random-forest.html)









