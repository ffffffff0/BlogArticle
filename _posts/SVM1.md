---
title: SVM-最大间隔分类
date: 2018-2-19
tags: [SVM, Python, MachineLearn]
mathjax: true
categories: [机器学习, SVM]
---

支持向量机属于监督学习算法，除了进行线性分类之外，SVM还可以使用核函数有效地进行非线性分类，将其输入隐式映射到高维特征空间中。
<!-- more -->

SVM分类器在几何形式上如下图：

![svmpic](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmpic.png)

在图中可以得知划分数据集的线存在很多条，紫色的分割线和绿色的分割线相比是不好的，在增加数据后紫色的分割线容易发生错误的分类。因此需要找到一条像绿色分割线这样的，让它离样本点都足够的大。

SVM分为三种：hard margin SVM,
            soft margin SVM,
            kernel SVM.
## 硬间隔(最大间隔分类器)
用数学的方法来描述最大间隔分类器。

假设样本点为: $\{(x_i, y_i)\mid_{i=1}^{N},x_i \in R ,y_i\in\{-1,1\}\}$.

则如果分类器分类正确满足:
$$s.t.\begin{cases}w^Tx_i+b>0, & \ y_i=+1\\\\
w^Tx_i+b<0, & \ y_i=-1\\\\
\end{cases}$$

如上的条件可以简化为：
$y_i(w^Tx_i+b)>0$

关于间隔margin，在图形上为上如图的d，也就是离分割平面最近的点的距离。

![svmpic2](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmpic2.png)

根据点到直线的距离公式：$d=|\frac{Ax_0+Bx_0+C}{\sqrt{A^2+B^2}}|$, 扩展到多维情况下为：$d=\frac{|w^Tx+b|}{\shortparallel w \shortparallel}$。

求解最小的d，即$margin=\min{d}=min_{w,b,x_i,i=1,...,N}\frac{|w^Tx_i+b|}{\shortparallel w \shortparallel}$。

最大化间隔为：
$$
\max{margin}=\max_{w,b}\min_{x_i,i=1,...N}\frac{|w^Tx_i+b|}{\shortparallel w \shortparallel}
$$
可以简化为：
$$
\max{margin}=\max_{w,b}\frac{1}{\shortparallel w \shortparallel}\min_{x_i,i=1,...,N}y_i(w^Tx_i+b)
$$
由于:
$$
exist R>0,y_i(w^Tx_i+b)>0,\Rightarrow \min_{x_i, i=1,...,N}y_i(w^Tx_i+b)=R
$$
假设R=1，故有$\max{margin}=\max_{w,b}\frac{1}{\shortparallel w \shortparallel}$，则求解问题转化为：
$$
\max_{w,b}\frac{1}{\shortparallel w \shortparallel}\\\\
s.t.\min{y_i(w^Tx_i+b)=1}
$$
转化为最小化问题：
$$
\min_{w,b}\frac{1}{2}w^Tw\\\\
s.t. y_i(w^Tx_i+b)\ge1, i=1,...,N
$$
### 求解模型

原问题为：
$$
\min_{w,b}\frac{1}{2}w^Tw\\\\
s.t. y_i(w^Tx_i+b)\ge1, i=1,...,N
$$
可以借助拉格朗日数乘法构造如下函数：
$$
\ell(w,b,\lambda)=\frac{1}{2}w^Tw+\sum_{i=1}^{N}\lambda_i(1-y_i(w^Tx_i+b))
$$
其中$\lambda_i$为拉格朗日乘子，满足$\lambda_i\ge0$, 结合上式可以得到如下：
$$
\begin{cases}
\min_{w,b} \max_{\lambda}{\ell(w,b,\lambda)}\\\\
s.t. \lambda_i\ge0
\end{cases}
$$
如下图所示可以证明上式与原问题等价。

![svmpic3](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmpic3.png)

我们的新目标函数，先求最大值，再求最小值。这样的话，我们首先就要面对带有需要求解的参数w和b的方程，而$\lambda$又是不等式约束，这个求解过程不好做。所以，我们需要使用拉格朗日函数对偶性，将最小和最大的位置交换一下，如下所示：
$$
\begin{cases}
\max_{\lambda} \min_{w,b} \ell(w,b,\lambda)\\\\
s.t.\lambda_i\ge0
\end{cases}
$$

求解 $\min_{w,b}\ell(w,b,\lambda)$:
$$
\frac{\partial \ell}{\partial w}=0;\\\\
\frac{\partial \ell}{\partial w}=\frac{1}{2}\cdot2\cdot w - \sum_{i=1}^{N}x_iy_i\lambda_i = 0;\\\\
w=\sum_{i=1}^{N}x_iy_i\lambda_i;
$$

$$
\frac{\partial \ell}{\partial d}=0;\\\\
\frac{\partial \ell}{\partial d}= -\sum_{i=1}^{N} \lambda_iy_i=0;
$$
将结果代入$\ell(w,b,\lambda)$中可以得到：
$$
\ell(w,b,\lambda)=\frac{1}{2}w^Tw-\sum_{i=1}^{N}\lambda_iy_i(w^T+b)+\sum_{i=1}^{N}\lambda_i
$$
$$
\Rightarrow \frac{1}{2}w^Tw+\sum_{i=1}^{N}\lambda_i-\sum_{i=1}^{N}\lambda_iy_iw^T-\sum_{i=1}^{N}\lambda_iy_ib
$$
$$
\Rightarrow \frac{1}{2}w^Tw+\sum_{i=1}^{N}\lambda_i-\sum_{i=1}^{N}\lambda_iy_iw^T
$$
$$
\Rightarrow \frac{1}{2}(\sum_{i=1}^{N}x_iy_i\lambda_i)^T(\sum_{i=1}^{N}x_iy_i\lambda_i)+\sum_{i=1}^{N}\lambda_i-\sum_{i=1}^{N}\lambda_iy_i(\sum_{i=1}^{N}x_iy_i\lambda_i)^T
$$
$$
\Rightarrow -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_i\lambda_jy_iy_jx_{i}^Tx_j + \sum_{i=1}^{N}\lambda_i
$$
可以看出，此时的$\ell(w,b,\lambda)$函数只含有一个变量，即$\lambda$。现在优化问题转化为如下的形式：

$$
\max_{\lambda} -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_i\lambda_jy_iy_jx_{i}^Tx_j + \sum_{i=1}^{N}\lambda_i;\\\\
s.t. \lambda_i\ge0, i=1,...,N;\\\\
\sum_{i=1}^{N} \lambda_iy_i=0;
$$

对于这个问题，我们有更高效的优化算法，即序列最小优化（SMO）算法。我们通过这个优化算法能得到$\lambda$，再根据$\lambda$，我们就可以求解出w和b，进而求得我们最初的目的：找到超平面。
