---
title: Django 数据库
date: 2016-6-8
tags: [Python, Django]
---
在Django中的MVC中， 视图用来描述要展现给用户的数据，
视图仅决定如何展现数据，而不是展现那些数据。

### 数据库配置
关于数据库的配置，这里使用mysql，python3使用mysqlclient库来链接，并更改settings中的设置。
<!-- more -->
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'Django',
        'USER': 'root',
        'PASSWORD': 'pwd',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```
在terminal中输入 python3 manage.py shell 来进行测试。

输入如下命令测试数据库的设置：

```python
from django.db import connection
cursor = connection.cursor()
```
如果没有错误信息，那么数据库配置正确。

### Django App
一个project可以包含很多个App及他们的配置。

一个app是一套Django功能的集合，通常包括模型和视图，按Python的包结构的方式存在。

系统对app有一个约定： 如果你使用了Django的数据库层（模型），你 必须创建一个Django app。 模型必须存放在apps中。

#### 创建App
```python
python3 manage.py startapp books
```
#### 创建模型
本例取自DjangoBook的例子

我们来假定下面的这些概念、字段和关系：

一个作者有姓，有名及email地址。

出版商有名称，地址，所在城市、省，国家，网站。

书籍有书名和出版日期。 它有一个或多个作者（和作者是多对多的关联关系[many-to-many]）， 只有一个出版商（和出版商是一对多的关联关系[one-to-many]，也被称作外键[foreign key]）

在models.py中输入：

```python
from django.db import models

class Publisher(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    state_province = models.CharField(max_length=30)
    country = models.CharField(max_length=50)
    website = models.URLField()

class Author(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=40)
    email = models.EmailField()

class Book(models.Model):
    title = models.CharField(max_length=100)
    authors = models.ManyToManyField(Author)
    publisher = models.ForeignKey(Publisher)
    publication_date = models.DateField()

```
相当于sql：

```python
CREATE TABLE "books_publisher" (
    "id" serial NOT NULL PRIMARY KEY,
    "name" varchar(30) NOT NULL,
    "address" varchar(50) NOT NULL,
    "city" varchar(60) NOT NULL,
    "state_province" varchar(30) NOT NULL,
    "country" varchar(50) NOT NULL,
    "website" varchar(200) NOT NULL
);
```
最后需要注意的是，我们并没有显式地为这些模型定义任何主键。 除非你单独指明，否则Django会自动为每个模型生成一个自增长的整数主键字段每个Django模型都要求有单独的主键。id

在settings中加入app后，验证模型的有效性：
```python
python3 manage.py makemigrations books
```
运行 makemigrations ，表示告诉Django你已经改变模型，你想储存这些改变。

如果模型没有问题的话，运行：
```python
python3 manage.py sqlmigrate books 001
```
生成sql语句， 命令并没有在数据库中真正创建数据表，只是把SQL语句段打印出来

也可以运行：
```python
python3 manage.py check
```
这将检查项目中的任何问题。不会改变模型。
```python
python3 manage.py migrate
```
这条命令将同步你的模型到数据库，并不能将模型的修改或删除同步到数据库；如果你修改或删除了一个模型，并想把它提交到数据库，migrate并不会做出任何处理。

### 模型内的增删改查
