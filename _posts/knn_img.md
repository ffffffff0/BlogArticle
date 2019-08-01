---
title: KNN 分类图片
date: 2019-4-17
tags: [Python, KNN,CS231N]
mathjax: true
categories: [机器学习, CS231N]
---
文章主要是对Stanford的CS231n的作业做一次记录。
> Q1: k-Nearest Neighbor classifier (20 points)
>The IPython Notebook knn.ipynb will walk you through implementing the kNN classifier.
<!--more-->

利用KNN算法来分类图片，图片数据集来自CIFAR10.
#### 导入数据集
数据集为二进制文件，需要做一下处理，分别返回训练集和测试集。

```
def unpickle(file):
    return pickle.load(file, encoding='latin1')
def loadCifarbatch(f):
    file = open(f, 'rb')
    datadict = unpickle(file)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y
def loadCifar10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_{}'.format(b))
        X, Y = loadCifarbatch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = loadCifarbatch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
```
#### 构建分类器
由于数据集数据很多，这里选择一部分。
```
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
```
计算图片之间的距离，函数中写了三个函数，分别是两次循环，一次循环，不写循环来求解距离矩阵，这里的距离采用欧式距离来计算。后来经过比对不写循环的效率最高。
```

class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, x, k=1, num_loop=0):

        if num_loop==0:
            dists = self.compute_distance_no_loop(x)
        elif num_loop==1:
            dists = self.compute_distance_one_loop(x)
        elif num_loop==2:
            dists = self.compute_distance_two_loop(x)
        else:
            raise ValueError('Invalid value %d for num_loop'%num_loop)
        
        return self.predict_labels(dists, k=k)
    

    def compute_distance_two_loop(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]

        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # 求解两张图片之间的欧式距离
                dists[i][j] = np.sqrt(np.sum((x[i]-self.x_train[j])**2))
        
        return dists
    

    def compute_distance_one_loop(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]

        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # 
            dists[i,:] = np.sqrt(np.sum((x[i]-self.x_train)**2, axis=1))
            # 利用了广播的功能来求欧式距离
            # sum(m, axis=1) 计算每行的和
            # sum(m, axis=0) 计算每列的和
        
        return dists
    
    
    def compute_distance_no_loop(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]

        # 即将欧式距离公式展开计算
        dists = np.zeros((num_test, num_train))
        dists += np.sum(self.x_train**2, axis=1).reshape(1, num_train)
        # 广播的应用 shape = (1, 5000)
        dists += np.sum(x**2, axis=1).reshape(num_test, 1)
        # 广播的应用 shape = (500, 1)
        dists -= 2*np.dot(x, self.x_train.T)
        dists = np.sqrt(dists)

        return dists
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            # 求最近的k个标签
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pre
```
#### 交叉验证
有时候，训练集数量较小（因此验证集的数量更小），人们会使用一种被称为交叉验证的方法，这种方法更加复杂些。还是用刚才的例子，如果是交叉验证集，我们就不是取1000个图像，而是将训练集平均分成5份，其中4份用来训练，1份用来验证。然后我们循环着取其中4份来训练，其中1份来验证，最后取所有5次验证结果的平均值作为算法验证结果。
![cross](https://image-1252432001.cos.ap-chengdu.myqcloud.com/blog/knn/cc88207c6c3c5e91df8b6367368f6450_hd.jpg)

迭代所有情况，分别求出准确率。
```
num_folds = 5
    k_choices = [1,3,5,8,10,12,15,20,50,100]

    x_train_folds = []
    y_train_folds = []

    # 分割训练集
    x_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_to_accuracies = {}

    classifier = KNearestNeighbor()
    for k in k_choices:
        accuracies = np.zeros(num_folds)
        for fold in range(num_folds):
            temp_x = x_train_folds[:]
            # 将x_train_folds[:] 赋予temp_x
            temp_y = y_train_folds[:]
            x_validate_fold = temp_x.pop(fold)
            # 构建验证集
            y_validate_fold = temp_y.pop(fold)

            temp_x = np.array([y for x in temp_x for y in x])
            # 构建数据集
            temp_y = np.array([y for x in temp_y for y in x])
            classifier.train(temp_x, temp_y)

            y_test_pred = classifier.predict(x_validate_fold, k=k)
            num_correct = np.sum(y_test_pred == y_validate_fold)
            accuracy = float(num_correct)/num_test
            accuracies[fold] = accuracy
        k_to_accuracies[k] = accuracies
    
    # 输出准确率
    for k in sorted(k_to_accuracies):
        for accuarcy in k_to_accuracies[k]:
            print('k=%d, accuracy=%f' % (k, accuracy))
    
```