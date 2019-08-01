---
title: 随机优化算法
date: 2019-4-9
tags: [Python, Optimization, Math]
mathjax: true
categories: Python
---

优化算法是尝试不同的解，并对不同的解进行评价来找到问题的最优解。

### 宿舍分配问题

有5间宿舍，每个宿舍有两个隔间，10个学生来竞争，每个学生都有一个首选和次选，来尝试寻找最优的分配。

```python
dorms = ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']
prefs=[('Toby', ('Bacchus', 'Hercules')),
       ('Steve', ('Zeus', 'Pluto')),
       ('Karen', ('Athena', 'Zeus')),
       ('Sarah', ('Zeus', 'Pluto')),
       ('Dave', ('Athena', 'Bacchus')), 
       ('Jeff', ('Hercules', 'Pluto')), 
       ('Fred', ('Pluto', 'Athena')), 
       ('Suzie', ('Bacchus', 'Hercules')), 
       ('Laura', ('Bacchus', 'Hercules')), 
       ('James', ('Hercules', 'Athena'))]
```

学生和宿舍如上；

<!-- more -->

首先要确定成本函数，才能寻找最优解；这里采用如果学生被安置的宿舍为首选宿舍，则成本为0；为次选宿舍则加1；不再选择之中，则加3(*构造成本函数时尽可能让最优解的成本为0*）；

```python
# 成本函数
def dormcost(vec):
    cost = 0
    slots = []
    for i in range(len(dorms)): slots+=[i, i]
    
    for i in range(len(vec)):
        x = int(vec[i])
        dorm = dorms[slots[x]]
        pref = prefs[i][1]

        if pref[0] == dorm: cost+=0
        elif pref[1]== dorm: cost+=1
        else: cost +=3
        
        del slots[x]
    
    return cost
```

### 随机优化算法

尝试使用四种算法来解决这个问题，分别为 随机搜索，爬山法，模拟退火算法，遗传算法；

##### 随机搜索

随机搜索算法顾名思义，随机产生1000次猜测，并对每一猜测的成本进行评价，找到最佳的猜测结果并返回。

```python
def randomoptimize(domain, costf):
    best = 9999999999999
    bestr = None
    for j in range(1000):
        r = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        # 随机产生一组序列来进行计算
        cost = costf(r)
        if cost <best:
            best = cost
            bestr = r
    return r
```

其中的**domain**为元组组成的列表，为每个变量的最大最小值；**costf** 为成本函数；

```python
domain = [(0, (len(dorms)*2)-i-1) for i in range(0, len(dorms)*2)]
# 建立如[(0,9), (0,8), (0,7),..., (0,0)]
```

但是由于随机的原因，得到的结果不尽相同；

##### 爬山法

随机的尝试各种解其实非常的低效，没有办法利用已经发现的优解。爬山法可以替代随机搜索法，爬山法以一个随机解出发，然后在解的周围来寻找更好的解，基本上就是贪心的策略；

```python
# 爬山法
def hillclimb(domain, costf):
    sol = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
    # 随机生成一组序列
    while 1:
        # 创建相邻的解
        neighbors = []
        for j in range(len(domain)):
            # 在原来的值上偏离一点
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:]) 
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost<best:
                best = cost
                sol = neighbors[j]
        if best==current:
            break
    return sol
```

爬山法比随机搜索更快，但是爬山法容易陷入局部最优解;

![局部最优](https://image-1252432001.cos.ap-chengdu.myqcloud.com/blog/optimization/timg.jpg)

##### 模拟退火算法

退火算法也以一个随机解开始，用一个变量来表示温度，这个温度开始时非常高，然后逐渐变低。在迭代过程中，算法会随机选择序列中的某个数字，然后朝着某个方向变化，如果新的序列成本变低，那么新的解变成当前的解，和爬山法很像，但是新的序列成本值变高也有可能成为当前解。这样可以避免局部最小值的一种尝试。模拟退火算法不仅接受一个更优的解，而且在退火过程中开始阶段接受表现较差的解。随着退火过程的不断进行，算法越来越不可能接受较差的解，直到最后，它只会接受更优的解。

其中接受的概率由下公式给出：
$$
p = e^{-\frac{highcost - lowcost}{temperature}}
$$

温度开始很高，指数接近0.所以概率几乎为1，随着温度的递减高成本和低成本之间的差异会变得重要。

```python

# 模拟退火算法
def annelintoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
    # 创建随机的序列

    while T>0.1:
        i = random.randint(0, len(domain)-1)
        # 随机选择一个索引值
        dirs = random.randint(-step, step)
        # 随机选择一个改变方向的索引值 
        vecb = vec[:]
        # 创建一个迭代的新列表
        vecb[i] += dirs
        # 改变其中的一个值
        if vecb[i]<domain[i][0]:                
            vecb[i] = domain[i][0]
        elif vecb[i]>domain[i][1]:
            vecb[i] = domain[i][1]
        # 应该是防止出界
        
        ea = costf(vec)
        eb = costf(vecb)
        # 计算新成本和旧成本

        if(eb<ea or random.random()<pow(math.e, -(eb-ea)/T)):
            vec = vecb
        # 比较是否为最好的解
        T = T*cool
    return vec
```

##### 遗传算法

遗传算法的过程是先随机生成一组解，叫做种群，迭代过程中，算法会计算整个种群的成本函数，得到一个列表，对列表进行排序，一个新的种群会被创建，这个新的种群包括排序后最顶端的一些序列，这一个过程叫做精英选拔，还有一些是经过变异or交叉操作形成的全新的解。

其中变异操作，是对序列的一些小的随机改变。如下：

> ```python
> [7, 5, 2, 3, 1, 6, 7, 1, 0, 3] ------->[7, 5, 2, 3, 1, 6, *5*, 1, 0, 3]
> ```

交叉操作是选取两个解，按照某种方式来进行剪切和组合来形成新的解。如下：

> ```python
> [*7, 5, 2, 3, 1, 6, 7*, 1, 0, 3]
> [7, 2, 2, 4, 5, 9, 1, *9, 4, 5*]     ------->  [7, 5, 2, 3, 1, 6, 7, 9, 4, 5]
> ```

```python
# 遗传算法
def geneticoptimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    
    # 变异操作
    def mutate(vec):
        i = random.randint(0, len(domain)-1)
        if random.random()<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]
    
    # 交叉操作
    def crossover(r1, r2):
        i = random.randint(1, len(domain)-2)
        return r1[0:i]+r2[i:]

    # 构造初始种群 
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)
    
    topelite = int(elite*popsize)
    # 胜出者的数量

    # print(pop)

    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        # 从小到大
        ranked = [v for (s, v) in scores]
        # 排序之后的序列
        pop = ranked[0: topelite]
        # 选取胜出的序列

        while len(pop)<popsize:
        # 添加变异和交叉之后的优胜者
            if random.random() < mutprob:
            # 变异和交叉的概率
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            # 变异
            else:
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))
            # 交叉
        print(scores[0][0])
    return scores[0][1]
```
————————————————————————————————
