---
title: SVM-编程实现算法
date: 2019-7-23
tags: [SVM, Python, MachineLearn]
mathjax: true
categories: [机器学习, SVM]
---
上篇SMO算法已推导完成，本篇在于核函数，编程实现一个完整的SVM分类过程。
<!--more-->
### 核函数

SVM处理线性可分的情况上面已经推导，而对于非线性的情况，由于线性不可分，SVM通过某种事先选择的非线性映射（核函数）将输入变量映到一个高维特征空间，将其变成在高维空间线性可分，在这个高维空间中寻找最优分类超平面。

在线性可分的情形下，超平面方程为：
$$
f(x)=\sum_{i=1}^n \alpha_iy_i\langle x_i, x \rangle +b
$$
线性不可分的情形下, 寻找一个非线性的映射到高维的特征空间。
$$
f(x)=\sum_{i=1}^n \alpha_iy_i\langle \Phi(x_i), \Phi(x) \rangle +b
$$
其中，$\Phi$为输入空间X到特征空间F的映射，核函数方法就是直接计算内积$\langle \Phi(x_i), \Phi(x) \rangle$，就如线性可分时的函数一样。

![示例](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/kernel.jpg)

对于图中的例子，通过变换将椭圆转为直线，将非线性分类问题变换为线性分类问题。

假设原空间为$\mathcal{X}\subset R^2, x=(x^{(1)}, x^{(2)})^T$, 新空间为$\mathcal{Z}\subset R^2, z=(z^{(1)},z^{(2)})^T$, 从原空间到新空间的变换(映射)为:
$$
z=\Phi(x)=((x^{(1)})^2, (x^{(2)})^2)^T
$$
经过变换，原空间中的点相应地变换为新空间中的点，原空间中的椭圆
$$
w_1(x^{(1)})^2+w_2(x^{(2)})^2+b=0
$$
变换为新空间中的直线
$$
w_1z^{(1)}+w_2z^{(2)}+b=0
$$
在新空间中直线$w_1z^{(1)}+w_2z^{(2)}+b=0$可以将变换后的正负实例点正确的分开，这样就解决了非线性不可分的问题。

>核函数的定义

设$\mathcal{X}$时输入空间(欧式空间$R^n$的子集或者离散集合)，又设$\mathcal{H}$为特征空间(希尔伯特空间)，如果存在一个从$\mathcal{X}$到$\mathcal{H}$的映射
$$
\phi(x):\mathcal{X}\rightarrow\mathcal{H}
$$
使得对所有$x, z\in\mathcal{X}$，函数$\mathcal{K}(x,z)$满足条件
$$
\mathcal{K}(x,z)=\phi(x)\cdot \phi(z)
$$
称$\mathcal{K}(x,z)$为核函数，$\phi(x)$为映射函数，其中$\phi(x)\cdot\phi(z)$为$\phi(x)$和$\phi(z)$的内积

核技巧就是在学习和预测中只定义核函数$\mathcal{K}(x,z)$，而不显式地定义映射函数$\phi$，通常情况下，直接计算$\mathcal{K}(x,z)$比较容易，而通过$\phi(x),\phi(z)$来计算$\mathcal{K}(x,z)$并不容易，$\phi$是输入空间$R^n$到特征空间$\mathcal{H}$的映射，特征空间$H$一般是高维空间，甚至为无穷维。可以看到，对于给定的核$\mathcal{K}(x,z)$，特征空间$H$和映射函数$\phi$的取法并不唯一，可以取不同的特征空间，即便是在同一特征空间里也可以取不同的映射。

核函数与映射的关系

>假设输入空间$R^2$，核函数是$K(x,z)=(x,z)^2$，试找出相关的特征空间$\mathcal{H}$和映射$\phi(x):R^2\rightarrow\mathcal{H}$

取特征空间$\mathcal{H}=R^3$, 记$x=(x^{(1)},x^{(2)})^T, z=(z^{(1)},z^{(2)})^T$, 由于
$$
(x\cdot z)=(x^{(1)}z^{(1)}+x^{(2)}z^{(2)})^2\\\\
\Rightarrow (x^{(1)}z^{(1)})^2+2x^{(1)}z^{(1)}x^{(2)}z^{(2)}+(x^{(2)}z^{(2)})^2
$$
所以可以取映射
$$
\phi(x)=((x^{(1)})^2, \sqrt2x^{(1)}x^{(2)}, (x^{(2)})^2)^T
$$
易证$\phi(x)\phi(z)=(x\cdot z)^2=\mathcal{K}(x,z)$

仍然取$\mathcal{H}=R^3$以及
$$
\phi(x)=\frac{1}{\sqrt2}((x^{(1)})^2-(x^{(2)})^2, 2x^{(1)}x^{(2)}, (x^{(1)})^2+(x^{(2)})^2)^T
$$
同样有$\phi(x)\phi(z)=(x\cdot z)^2=\mathcal{K}(x,z)$， 故映射不唯一

### SVM手写体识别

首先导入第三方库，建立数据结构

```

import matplotlib.pyplot as plt
import numpy as np
import random

class optStruct:

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        数据结构，维护所有需要操作的值
        Parameters：
            dataMatIn - 数据矩阵
            classLabels - 数据标签
            C - 松弛变量
            toler - 容错率
            kTup - 包含核函数信息的元组,第一个参数存放核函数类别，
            第二个参数存放必要的核函数需要用到的参数
        """
        # 数据矩阵
        self.X = dataMatIn
        # 数据标签
        self.labelMat = classLabels
        # 松弛变量
        self.C = C
        # 容错率
        self.tol = toler
        # 数据矩阵的行数
        self.m = np.shape(dataMatIn)[0]
        # 初始化alphas参数为0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 初始化b参数为0
        self.b = 0
        # 初始化误差缓存,第一列为是否有效的标志位,第二列为实际的误差E的值。
        self.eCache = np.mat(np.zeros((self.m, 2)))
        # 初始化核
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 计算所有数据的核
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.x, self.x[i, :], kTup)
```

核函数的应用和实现
```
def kernelTrans(X, A, kTup):
    """
    通过核函数将数据转换更高维的空间
    参数:
        X: 数据矩阵
        A: 单个数据的向量
        kTup: 包含核函数信息的元组
    返回:
        K: 计算的核k
    """
    m, _ = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        # 线性核函数, 只进行内积
        K = X * A.T
    elif kTup[0] == 'rbf':
        # 高斯核函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]** 2))
    else:
        raise NameError('核函数无法识别')

    return K
```

读取数据

```

def loadDataSet(fileName):
    """
    读取数据
    参数:
        fileName: 文件名
    返回:
        dataMat: 数据矩阵
        labelMat: 数据标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
```

误差计算

```

def calcEk(ops, k):
    """
    计算误差
    参数：
        ops: 数据结构
        k: 标号为k的数据
    返回：
        Ek: 标号为k的数据误差
    """
    fxk = float(np.multiply(ops.alphas, ops.labelMat).T * ops.K[:, k] + ops.b)
    Ek = fxk - float(ops.labelMat[k])
    return Ek
```

启发式选择

```
def selectJrand(i, m):
    """
    随机选择alpha_j 的索引
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def selectJ(i, ops, Ei):
    """
    内循环启发方式
    参数:
        i: 标号为i的数据的索引值
        ops: 数据结构
        Ei: 标号为i的数据误差
    返回:
        j,maxK: 标号为j或maxK的数据索引值
        Ej: 标号为j的数据误差
    """
    # 初始化
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 根据Ei更新误差
    ops.eCache[i] = [1, Ei]
    # 返回误差不为0的数据的索引值
    validEcacheList = np.nonzero(ops.eCache[:, 0].A)[0]
    # 有不为0的误差
    if (len(validEcacheList)) > 1:
        # 遍历找到最大的Ek
        for k in validEcacheList:
            # 不计算i, 浪费时间
            if k == i:
                continue
            # 计算Ek
            Ek = calcEk(ops, k)
            # 计算|Ei-Ek|
            deltaE = abs(Ei - Ek)
            # 找到最大的DeltaE
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        # 返回maxk, Ej
        return maxK, Ej
    # 没有不为0的误差
    else:
        # 随机选择j
        j = selectJrand(i, ops.m)
        # 计算Ej
        Ej = calcEk(ops, j)
    return j, Ej
```

更新误差

```
def updateEk(ops, k):
    """
    计算Ek, 并更新误差缓存
    """
    # 计算Ek
    Ek = calcEk(ops, k)
    # 更新误差缓存
    ops.eCache[k] = [1, Ek]
```

修剪$\alpha_j$

```
def clipAlpha(aj, H, L):
    """
    修剪alpha
    参数:
        aj: alphaj的值
        H: alpha上限
        L: alpha下限
    返回:
        aj: 修剪后alphaj的值 
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
```

图片转向量

```
def img2vector(filename):
    """
    将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    """
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

加载图像

```
def loadImages(dirName):
    """
    加载图片
    Parameters:
        dirName - 文件夹的名字
    Returns:
        trainingMat - 数据矩阵
        hwLabels - 数据标签
    """
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels
```

优化的SMO算法

```
def innerL(i, oS):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    #步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    #优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #使用内循环启发方式2选择alpha_j,并计算Ej
        j,Ej = selectJ(i, oS, Ei)
        #保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        #步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        #步骤3：计算eta
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        #步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        #步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS, i)
        #步骤7：更新b_1和b_2
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        #步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0
```

优化后完整的SMO算法

```
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
    """
    完整的线性SMO算法
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
        maxIter - 最大迭代次数
        kTup - 包含核函数信息的元组
    Returns:
        oS.b - SMO算法计算的b
        oS.alphas - SMO算法计算的alphas
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)                #初始化数据结构
    iter = 0                                                                                         #初始化当前迭代次数
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):                            #遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:                                               #遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)                                                    #使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:                                                                                         #遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]                        #遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:                                                                                #遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):                                                                #如果alpha没有更新,计算全样本遍历
            entireSet = True 
        print("迭代次数: %d" % iter)
    return oS.b,oS.alphas
```

手写体识别的测试函数

```
def testDigits(kTup=('rbf', 10)):
    """
    测试函数
    Parameters:
        kTup - 包含核函数信息的元组
    Returns:
        无
    """
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("测试集错误率: %.2f%%" % (float(errorCount)/m))
```

运行结果为
![svmresult](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmresult.png)
