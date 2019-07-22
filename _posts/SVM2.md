---
title: SVM-SMO算法
date: 2019-7-19
tags: [SVM, Python, MachineLearn]
mathjax: true
---
上篇已经得到了可以用SMO算法求解的目标函数，此篇用于梳理SMO算法，编写程序实现SMO算法。
<!--more-->

### 松弛变量

将最大值问题转换成最小值问题，目标函数为：
$$
\begin{aligned}
\min_{\lambda} &\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_i\lambda_jy_iy_jx_{i}^Tx_j - \sum_{i=1}^{N}\lambda_i;\\\\
s.t.&\lambda_i\ge0, i=1,...,N;\\\\&
\sum_{i=1}^{N} \lambda_iy_i=0;
\end{aligned}
$$

>对于上述目标函数，是存在一个假设的，即数据100%线性可分。但是，几乎所有数据都不那么干净。对于这种偏离正常位置很远的数据点，在我们原来的 SVM 模型里，异常点的存在有可能造成很大的影响，因为超平面本身就是只有少数几个 support vector 组成的，如果这些 support vector 里又存在异常点的话，其影响就很大了。

![svmpic4](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svmpic4.png)

>用黑圈圈起来的那个蓝点是一个 outlier ，它偏离了自己原本所应该在的那个半空间，如果直接忽略掉它的话，原来的分隔超平面还是挺好的，但是由于这个 outlier 的出现，导致分隔超平面不得不被挤歪了，变成途中黑色虚线所示（这只是一个示意图，并没有严格计算精确坐标），同时 margin 也相应变小了。当然，更严重的情况是，如果这个 outlier 再往右上移动一些距离的话，我们将无法构造出能将数据分开的超平面来。

为了应对这样的情况，需要允许分类的数据点偏离一定的距离，即：
$$
y_i(w^Tx_i+b)\ge1-\xi_i, i=1,...,N
$$
其中$\xi_i$称为松弛变量，对应数据点$\xi_i$允许偏离的的量。如果我们运行 ξi 任意大的话，那任意的超平面都是符合条件的了。所以，我们在原来的目标函数后面加上一项，使得这些$\xi_i$的总和也要最小， 即：
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
\max_\alpha \sum_{i=1}^n\alpha_i – \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle
$$
由于得到$C-\alpha_i-r_i=0$，而且$r_i\geq 0$，故$\alpha_i\leq C$，则对偶问题为：
$$
\begin{cases} 
\min_\alpha & \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle - \sum_{i=1}^n\alpha_i \\\\ 
s.t., &0\leq \alpha_i\leq C, i=1,\ldots,n \\\\ 
&\sum_{i=1}^n\alpha_iy_i = 0 
\end{cases}
$$

### 求解方法
对于上面的对偶问题，其中$\alpha$变量来说，这是一个二次函数，优化$\alpha$非常容易，凸优化梯度下降一定能找到最优。

坐标下降(Coordinate Descend)的变种，它每次只选择一个维度，例如 $\alpha=(\alpha_1,…,\alpha_n)$，它每次选取$\alpha_i$为变量，而将 $\alpha_1,…,\alpha_{i−1},\alpha_{i+1},…,\alpha_n$都看成是常量，从而原始的问题在这一步变成一个一元函数，然后针对这个一元函数求最小值，如此反复轮换不同的维度进行迭代，将复杂的问题简单化。

在考虑约束条件的情况下，这问题中约束条件$\sum_{i=1}^n\alpha_iy_i = 0$，很明显选取一个变量是无用的。在SMO算法中，一次选取两个变量进行优化，推导可得：
$$
\mathcal{W(\alpha_1, \alpha_2)}=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle - \sum_{i=1}^n\alpha_i
$$
$$
\Rightarrow \frac{1}{2}\sum_{i=1}^n\left(\sum_{j=1}^2\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle+\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle\right)-\alpha_1-\alpha_2-\sum_{i=3}^n\alpha_i
$$
$$
\begin{aligned}
&\Rightarrow \frac{1}{2}\sum_{i=1}^2\left(\sum_{j=1}^2\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle+\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle\right)\\\\
&+\frac{1}{2}\sum_{i=3}^n\left(\sum_{j=1}^2\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle+\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle\right)-\alpha_1-\alpha_2-\sum_{i=3}^n\alpha_i
\end{aligned}
$$
$$
\begin{aligned}
&\Rightarrow \frac{1}{2}\sum_{i=1}^2\sum_{j=1}^2\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle+\sum_{i=1}^2\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle\\\\&+\frac{1}{2}\sum_{i=3}^n\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle-\alpha_1-\alpha_2-\sum_{i=3}^n\alpha_i
\end{aligned}
$$
$$
\begin{aligned}
&\Rightarrow \frac{1}{2}\alpha_{1}^2\langle x_1,x_1\rangle+\frac{1}{2}\alpha_{1}^2\langle x_2,x_2 \rangle+y_1y_2\alpha_1\alpha_2\langle x_1, x_2\rangle+\alpha_1y_1\sum_{j=3}^ny_j\alpha_j\langle x_1,x_j\rangle\\\\&+\alpha_2y_2\sum_{j=3}^ny_j\alpha_j\langle x_2,x_j\rangle+\frac{1}{2}\sum_{i=3}^n\sum_{j=3}^n\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle-\alpha_1-\alpha_2-\sum_{i=3}^n\alpha_i
\end{aligned}
$$
为了描述方便，定义符号如下：$K_{ij}=\langle x_i,x_j\rangle$，目标函数可写为(常数可省略)：
$$
\begin{aligned}
\min_{\alpha_1,\alpha_2} \mathcal{W(\alpha_1,\alpha_2)}=&\frac{1}{2}\alpha_{1}^2K_{11}+\frac{1}{2}\alpha_{1}^2K_{22}+y_1y_2\alpha_1\alpha_2K_{12}+\alpha_1y_1\sum_{j=3}^ny_j\alpha_jK_{j1}\\\\&+\alpha_2y_2\sum_{j=3}^ny_j\alpha_jK_{j2}-(\alpha_1+\alpha_2)+Constant
\end{aligned}
$$
约束条件为：
$$
s.t.\qquad\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^ny_i\alpha_i=B\\\\
0\le\alpha_i\le C,i=1,2
$$
由于只有两个变量($\alpha_1,\alpha_2$)，约束条件可在二维空间中图像表示为：
![svm5](https://image-1252432001.cos.ap-chengdu.myqcloud.com/SVM/svm5.jpg)

假设问题的初始可行解为$(\alpha_{1}^{old},\alpha_{2}^{old})$，最优解为$(\alpha_{1}^{new},\alpha_{2}^{new})$。因为两个因子不好同时求解，所以可以先求第二个乘子$\alpha_2$的解$(\alpha_{2}^{new})$，得到$\alpha_2$的解$(\alpha_{2}^{new})$之后，再用$\alpha_2$的解$(\alpha_{2}^{new})$表示$\alpha_1$的解$(\alpha_{1}^{new})$。为了求解$\alpha_{2}^{new}$，得先确定$\alpha_{2}^{new}$的取值范围。

由于$\alpha_{2}^{new}$满足约束条件，即它满足如下条件：
$$
L\le\alpha_{2}^{new}\le H
$$
其中，L和H是$\alpha_{2}^{new}$所在对角线端点的界，如果$y_1\ne y_2$，则有
$$
L=\max(0,\alpha_{2}^{old}-\alpha_{1}^{old}),H=min(C,C+\alpha_{2}^{old}-\alpha_{1}^{old})
$$
如果$y_1=y_2$，则有
$$
L=\max(0,\alpha_{2}^{old}+\alpha_{1}^{old}-C),H=min(C,\alpha_{2}^{old}+\alpha_{1}^{old})
$$
首先求未考虑约束$0\le\alpha_i \le C, i=1,2$时的$\alpha_2$的最优解$\alpha_{2}^{unew}$;然后再求解考虑此条件的解$\alpha_{2}^{new}$。为了叙述简单，记
$$
g(x)=\sum_{i=1}^N \alpha_iy_i\langle x_i, x\rangle +b\\\\
v_i=\sum_{j=3}^N\alpha_iy_iK_{ij}=g(x_i)-\sum_{j=1}^2\alpha_jy_jK_{ij}-b,\quad i=1,2
$$
目标函数又可写为
$$
\mathcal{W(\alpha_1,\alpha_2)}=\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2)+y_1v_1\alpha_1+y_2v_2\alpha_2
$$
其中常数部分省略，因为对于$\alpha_1$和$\alpha_2$来说，它们都是常数项，在求导的时候，直接变为0。对于这个目标函数，如果对其求导，还有个未知数$\alpha_1$，所以要推导出$\alpha_1$和$\alpha_2$的关系，然后用$\alpha_2$代替$\alpha_1$，这样目标函数就剩一个未知数了，可以利用约束条件来得到
$$
\sum_{i=1}^n\alpha_iy_i=0\\\\
\Rightarrow\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^n\alpha_iy_i=B\\\\
\Rightarrow\alpha_1=\frac{1}{y_1}\left(B-\alpha_2y_2\right)\\\\
\Rightarrow\alpha_1=y_1\left(B-\alpha_2y_2\right)
$$
代入目标函数可得
$$
\begin{aligned}
\mathcal{W(\alpha_2)}=&\frac{1}{2}K_{11}(B-\alpha_2y_2)^2+\frac{1}{2}K_{22}\alpha_2^2+y_2K_{12}(B-\alpha_2y_2)\alpha_2\\\\&-(y_1(B-\alpha_2y_2)+\alpha_2)+v_1(B-\alpha_2y_2)+y_2v_2\alpha_2
\end{aligned}
$$
对$\alpha_2$求导数：
$$
\frac{\partial W}{\partial \alpha_2}=K_{11}\alpha_2+K_{22}\alpha_2-2K_{12}\alpha_2-k_{11}By_2+K_{12}By_2+y_1y_2-1-v_1y_2+y_2v_2
$$
令其为0，可得：
$$
\begin{aligned}
(K_{11}+K_{22}-2K_{12})\alpha_2=&y_2(y_2-y_1+BK_{11}-BK_{12}+v_1-v_2)\\\\
=&y_2\lbrack y_2-y_1+BK_{11}-BK_{12}+\lgroup g(x_1)-\sum_{j=1}^2\alpha_jy_jK_{1j}-b\rgroup\\\\
&-\lgroup g(x_2)-\sum_{j=1}^2\alpha_jy_jK_{2j}-b\rgroup]
\end{aligned}
$$
将$\alpha_1^{old}y_1+\alpha_2^{old}y_2=B$代入，得到
$$
\begin{aligned}
(K_{11}+K_{22}-2K_{12})\alpha_2^{unew}=&y_2[y_2-y_1+(\alpha_1^{old}y_1+\alpha_2^{old}y_2)K_{11}-(\alpha_1^{old}y_1+\alpha_2^{old}y_2)K_{12}\\\\&+\lgroup g(x_1)-\sum_{j=1}^2\alpha_jy_jK_{1j}-b\rgroup
-\lgroup g(x_2)-\sum_{j=1}^2\alpha_jy_jK_{2j}-b\rgroup]\\\\&=
y_2[(K_{11}+K_{22}-2K_{12})\alpha_2^{old}y_2+y_2-y_1+g(x_1)-g(x_2)]
\end{aligned}
$$
令
$$
E_i=g(x_i)-y_i=\left(\sum_{i=1}^N \alpha_iy_i\langle x_i, x\rangle +b\right)-y_i,\qquad i=1,2
$$
$E_i$为函数$g(x)$对输入$x_i$的预测值和真实值输出$y_i$之差。则上式又可简化为
$$
(K_{11}+K_{22}-2K_{12})\alpha_2^{unew}=(K_{11}+K_{22}-2K_{12})\alpha_2^{old}+y_2(E_1-E_2)
$$
令$\eta=K_{11}+K_{22}-2K_{12}$代入，可以得到
$$
\alpha_2^{unew}=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{\eta}
$$
要满足不等式约束必须将$\alpha_2^{unew}$限制在区间$[L,H]$内，可得：
$$
\alpha_2^{new}=
\begin{cases}
H,\quad\alpha_2^{unew}>H\\\\
\alpha_2^{unew},\quad L\le\alpha_2^{unew}\le H\\\\
L,\quad\alpha_2^{unew}<H
\end{cases}
$$
可以利用$\alpha_2^{new}$求得$\alpha_1^{new}$为
$$
\alpha_1^{old}y_1+\alpha_2^{old}y_2=\alpha_1^{new}y_1+\alpha_2^{new}=B\\\\
\Rightarrow\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})
$$
于是得到最优化问题的解$(\alpha_1^{new}, \alpha_2^{new})$。

### 