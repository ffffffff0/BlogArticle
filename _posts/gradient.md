---
title: 梯度下降
date: 2019-6-1
tags: [Python, Gradient, CS231N]
mathjax: true
---
## 什么是梯度

一元函数的导数就是它的梯度，在多元函数中偏导就是它的梯度。

如一元函数$f(x) = (x-a)^2$的导数是$\frac{df}{dx}=2(x-a)$

对一元函数的求导公式如下：
$$\displaystyle\frac{df(x)}{dx}=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$$

有二元函数$f(x_1,x_2)=x_1^2+x_1x_2-3x_2$.
那么它的梯度为：
$$\nabla f=(\frac{\partial f}{\partial x_1 }, \frac{\partial f}{\partial x_2})=(2x_1+x_2, x_1-3)$$

如果取在点(1,1)的梯度值，则为(3, -2)

<!-- more -->
## 什么是梯度下降

在凸优化问题中，局部的最优解一定是全局最优解，也就是说在凸优化问题中导数为0(梯度为0向量)为最优解。

梯度的方向是函数增长速度最快的方向，那么梯度的反方向就是函数减少最快的方向。
为了寻找最优解，我们可以沿着梯度的反方向进行线性搜索，每次搜索的步长设为$\alpha$，直到梯度与0向量接近。这就是梯度下降方法。

算法步骤如下：

- (1) i=0时，设置初始点$x^0 = (x_1^0,x_2^0,...,x_n^0)$，设置步长也称学习率$\alpha$，设置迭代的误差阈值$tol$
- (2) 计算目标函数在x^i上的梯度$\nabla fx^i$
  
- (3) 计算${x}^{i+1}$, 公式如下：

$${x}^{i+1} = x^i - \alpha \nabla f x^i$$

- (4) 计算梯度$\nabla f{x}^{i+1}$，如果$\|\nabla f_{\textbf{x}^{i+1}}\|_2\leq tol$则迭代停止，则最优解为${x}^{i+1}$，如果梯度的二范数大于$tol$，那么i=i+1，并返回(3)。

## full-batch gradient descent

我们通常会有一个损失函数$L(\beta)$，其中向量$β=(β0,β1,⋯,βn)$是模型中的参数，我们需要找到最优的$\beta$来最小化损失函数$L(\beta)$。所以说，模型的训练过程也就是寻找最优解的过程。

利用一个简单的回归问题为例，自变量x，预测变量y，模型如下：
$$y=\beta_0 + \beta_1 x$$

$\beta(\beta_0, \beta_1)$为回归系数，最小二乘法的函数$L(\beta)$为损失函数:
$$L(\beta)=\frac{1}{N}\sum_{j=1}^{N} (y_j-\hat y_j)^2=\sum_{j=1}^N \frac{1}{N}(\beta_0+\beta_1 x_j - \hat y_j)^2$$

其中$\hat y_j$为第j个样本的真实值，$y_j$为根据回归系数预测的第j个样本的预测值。

计算损失函数的梯度：
$$\nabla L(\beta)=(\frac{\partial L}{\partial \beta_0}, \frac{\partial L}{\partial \beta_1})=(\frac{2}{N}\sum_{j=1}^{N} (\beta_0 + \beta_1 x_j - \hat y_j), \frac{2}{N}\sum_{j=1}^{N} x_j(\beta_0 + \beta_1 x_j - y_j))$$

重复梯度下降的算法步骤可以寻找到最优的回归系数。

利用真实数据进行梯度下降的代码实现：

```
import numpy as np
import pandas as pd

# 导入数据
######
# 初始设置
# 初始值
beta = [1, 1]
# 学习率
alpha = 0.2
# 阈值
tol_L = 0.1

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

# 定义计算梯度的函数
def compute_grad(beta, x, y):
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

# 定义更新beta的函数
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# 定义计算RMSE的函数
# RMSE-->root of mean square error
# root of mean squared error
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# 进行第一次计算
grad = compute_grad(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x, y)
    loss = loss_new
    loss_new = rmse(beta, x, y)
    i += 1
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))

```

## 随机梯度下降

随机梯度下降法(Stochastic Gradient Decent, SGD)是对普通梯度下降法计算效率的改进算法。本质上来说，我们预期随机梯度下降法得到的结果和普通梯度下降法相接近；SGD的优势是更快地计算梯度。

在full-batch gradient descent中梯度计算
$$\nabla L(\beta)=(\frac{\partial L}{\partial \beta_0}, \frac{\partial L}{\partial \beta_1})=(\frac{2}{N}\sum_{j=1}^{N} (\beta_0 + \beta_1 x_j - \hat y_j), \frac{2}{N}\sum_{j=1}^{N} x_j(\beta_0 + \beta_1 x_j - y_j))$$

可以看出计算一次梯度的代价很大，随机梯度下降可以降低计算梯度的代价。也即是说SGD适合在大样本的训练。

SGD在计算梯度时，并不使用全部的样本，而是随机选取一个样本$(x_r, \hat y_r)$。
$$\nabla L(\beta) = (\frac{\partial L}{\partial \beta_0 }, \frac{\partial L}{\partial \beta_1})=(2(\beta_0 + \beta_1 x_r -\hat{y_r}), 2x_r(\beta_0 + \beta_1x_r - \hat{y_r})$$

上面代码中计算梯度可以更改为：

```
def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    r = np.random.randint(0, len(x))
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)

```
SGD尽管加快了每次迭代的计算速度，但是也带了收敛不稳定的缺陷。
全批量梯度下降虽然稳定，但速度较慢；SGD虽然快，但是不够稳定。

## Mini-batch Stochastic Gradient Decent

Mini-batch Stochastic Gradient Decent 是对上面两种方法身的中和的产物。

Mini-batch Stochastic Gradient Decent的关键是不同于其他两种方法，使用b个随机不同的样本来计算梯度
$$\nabla L(\beta) = (\frac{\partial L}{\partial \beta_0}, \frac{\partial L}{\partial \beta_1})=(\frac{2}{b}\sum_{j=1}^{b} (\beta_0 + \beta_1x_j-\hat{y_j}), \frac{2}{b}\sum_{j=1}^{b}x_j(\beta_0 + \beta_1x_j-\hat{y_j})$$

可以看出当b=1时，此时与随机梯度下降相同，当b=N时，此时与
full-batch gradient descent相同。参数b的选择会影响方法的实现，应该根据样本的大小来确定b的值。

计算梯度的函数代码为：
```
def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)
```

## SVM 损失函数的梯度推导

SVM的损失函数想要SVM在正确分类上的得分始终比不正确分类上的得分高出一个边界值 $\Delta$。放在线性分类其中就是，第i个数据中包含图像$x_i$的像素和代表正确类别的标签$y_i$。评分函数输入像素数据，然后通过公式$f(x_i,W)$来计算不同分类类别的分值。这里我们将分值简写为s。比如，针对第j个类别的得分就是第j个元素：$s_j=f(x_i,W)_j$。针对第i个数据的多类SVM的损失函数定义如下：

$$\displaystyle L_i=\sum_{j\not=y_i}max(0,s_j-s_{y_i}+\Delta)$$
利用线性评分函数$(f(x_i,W)=Wx_i)$，可以将损失函数的公式改写为：
$$\displaystyle L_i=\sum_{j\not=y_i}max(0,w^T_jx_i-w^T_{y_i}x_i+\Delta)$$

>$x_i$示第i个样本,为行向量。假设有N个样本,特征个数为D

>$y_j$表示该样本的label，假设有C个类

>Δ是margin

>$w_j$为第j个类的权重，为长度为D的列向量。

>$w_j$为我们要学习的参数，总共有C∗D 个，用W表示.

损失函数$L_i$关于W的梯度可以表示为：
$$\frac{\partial{L_i}}{\partial{w}}=[\frac{d{L_i}}{d{w_1}},\frac{d{L_i}}{d{w_2}},...,\frac{d{L_i}}{d{w_c}}]=\left( \begin{array}{ccc}\frac{d{L_i}}{d{w_{11}}} & \frac{d{L_i}}{d{w_{12}}} & \ldots & \frac{d{L_i}}{d{w_{c1}}}\\\vdots & \vdots & \ddots & \vdots \\\frac{d{L_i}}{d{w_{1d}}} & \frac{d{L_i}}{d{w_{2d}}} & \ldots & \frac{d{L_i}}{d{w_{cd}}}\end{array}\right)$$

分析矩阵的一个元素：$\frac{d{L_i}}{d{w_{11}}}$
其中：
$$
\begin{aligned}
Li=&max(0,xi1w11+xi2w12…+xiDw1D−xi1wyi1−xi2wyi2…\\\\&−xiDwyiD+Δ)+max(0,xi1w21+xi2w22…+xiDw2D\\\\&−xi1wyi1−xi2wyi2…−xiDwyiD+Δ)+⋮max(0,xi1wC1\\\\&+xi2wC2…+xiDwCD−xi1wyi1−xi2wyi2…−xiDwyiD+Δ)
\end{aligned}
$$


如果 
$$w^T_1xi−w^T_{yi}xi+Δ>0$$

那么有
$$\frac{dL_i}{dw_{11}}=x_{i1}$$

借助指示函数，可以表示为
$$\frac{dL_i}{dw_{11}}=1(w^T_1xi−w^T_{yi}xi+Δ>0)x_{i1}$$

类似可得：
$$
\frac{dL_i}{dw_{12}} = \mathbb{1}(w_1^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_{i2} \\\\\frac{dL_i}{dw_{13}} = \mathbb{1}(w_1^Tx_i - w_{y_i}^Tx_i +       \Delta > 0) x_{i3} \\\\\vdots \\\\               \frac{dL_i}{dw_{1D}} = \mathbb{1}(w_1^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_{iD}
$$
即得：
$$\frac{dL_i}{dw_{j}} = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0)\begin{bmatrix}
x_{i1} \\\\ x_{i2} \\\\ \vdots \\\\ x_{iD} \end{bmatrix}\\\\
= \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i^T
$$
也就是：
$$\frac{dL_i}{dw_{y_i}} = - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)
  \begin{bmatrix}
  x_{i1} \\\\
  x_{i2} \\\\
  \vdots \\\\
  x_{iD}
  \end{bmatrix}
\\\\
= - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0) x_i^T$$
end