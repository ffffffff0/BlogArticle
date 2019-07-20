---
title: SVM-SMO算法
date: 2019-7-19
tags: [SVM, Python, MachineLearn]
mathjax: true
---
上篇已经得到了可以用SMO算法求解的目标函数，此篇用于梳理SMO算法，编写程序实现SMO算法。
<!--more-->
### SMO算法

将最大值问题转换成最小值问题，目标函数为：
$$
\min_{\lambda} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_i\lambda_jy_iy_jx_{i}^Tx_j - \sum_{i=1}^{N}\lambda_i;\\\\
s.t. \lambda_i\ge0, i=1,...,N;\\\\
\sum_{i=1}^{N} \lambda_iy_i=0;
$$

>对于上述目标函数，是存在一个假设的，即数据100%线性可分。但是，几乎所有数据都不那么干净。对于这种偏离正常位置很远的数据点，在我们原来的 SVM 模型里，异常点的存在有可能造成很大的影响，因为超平面本身就是只有少数几个 support vector 组成的，如果这些 support vector 里又存在异常点的话，其影响就很大了。

![svmpic4](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmpic4.png)

>用黑圈圈起来的那个蓝点是一个 outlier ，它偏离了自己原本所应该在的那个半空间，如果直接忽略掉它的话，原来的分隔超平面还是挺好的，但是由于这个 outlier 的出现，导致分隔超平面不得不被挤歪了，变成途中黑色虚线所示（这只是一个示意图，并没有严格计算精确坐标），同时 margin 也相应变小了。当然，更严重的情况是，如果这个 outlier 再往右上移动一些距离的话，我们将无法构造出能将数据分开的超平面来。

为了应对这样的情况，需要允许分类的数据点偏离一定的距离，即：
$$
y_i(w^Tx_i+b)\ge1-\xi_i, i=1,...,N
$$
其中$\xi_i$称为松弛变量，对应数据点 xi 允许偏离的的量。如果我们运行 ξi 任意大的话，那任意的超平面都是符合条件的了。所以，我们在原来的目标函数后面加上一项，使得这些$\xi_i$的总和也要最小， 即：
$$
\min \frac{1}{2}\|w\|^2\color{red}{+C\sum_{i=1}^n \xi_i}
$$
其中C是一个参数，用于控制目标函数中两项（“寻找 margin 最大的超平面”和“保证数据点偏差量最小”）之间的权重。其中$\xi$是需要优化的变量，而C是一个事先确定好的常量。综上可得：

$$
\begin{cases} 
\min & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n\xi_i \\\\
s.t., & y_i(w^Tx_i+b)\geq 1-\xi_i, i=1,\ldots,n \\\\
& \xi_i \geq 0, i=1,\ldots,n 
\end{cases}
$$
仍然使用拉格朗日数乘法：
$$
\mathcal{L}(w,b,\xi,\alpha,r)=\frac{1}{2}\|w\|^2 + C\sum_{i=1}^n\xi_i – \sum_{i=1}^n\alpha_i \left(y_i(w^Tx_i+b)-1+\xi_i\right) – \sum_{i=1}^n r_i\xi_i
$$
同理，用$\mathcal{L}$对$w,b,\xi_i$求最小化。
$$
\begin{cases} 
\frac{\partial \mathcal{L}}{\partial w}=0 &\Rightarrow w=\sum_{i=1}^n \alpha_i y_i x_i \\\\
\frac{\partial \mathcal{L}}{\partial b} = 0 &\Rightarrow \sum_{i=1}^n \alpha_i y_i = 0 \\\\
\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 &\Rightarrow C-\alpha_i-r_i=0, \quad i=1,\ldots,n 
\end{cases}
$$
将$w$带入$\mathcal{L}$化简可得：
$$
\max_\alpha \sum_{i=1}^n\alpha_i – \frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle
$$
由于得到$C-\alpha_i-r_i=0$，而且$r_i\geq 0$，故$\alpha_i\leq C$，则对偶问题为：
$$
\begin{cases} 
\min_\alpha & \frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle - \sum_{i=1}^n\alpha_i \\\\ 
s.t., &0\leq \alpha_i\leq C, i=1,\ldots,n \\\\ 
&\sum_{i=1}^n\alpha_iy_i = 0 
\end{cases}
$$
