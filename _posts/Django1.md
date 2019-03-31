---
title: Django MTV模型
date: 2016-6-6
tags: [Python, Django]
---

### MTV 模型
简单的向服务器发送一个requests，服务器会返回一个responses。

具体的操作为：

发送一个请求， requests 会到views区， 找到特定的views， views到models层查找数据， models为托管数据的层级和一些对数据进行操作， 接下来会将找到的数据到templates层，templates来展示数据。这就是简单的 MTV 模型
<!-- more -->
Django 里更关注的是模型（Model）、模板(Template)和视图（Views），Django 也被称为 MTV 框架 。在 MTV 开发模式中：

M 代表模型（Model），即数据存取层。 该层处理与数据相关的所有事务： 如何存取、如何验证有效性、包含哪些行为以及数据之间的关系等。

T 代表模板(Template)，即表现层。 该层处理与表现相关的决定： 如何在页面或其他类型文档中进行显示。

V 代表视图（View），即业务逻辑层。 该层包含存取模型及调取恰当模板的相关逻辑。 你可以把它看作模型与模板之间的桥梁。

### 创建 Django 项目
使用pycharm中的terminal：

```python
python3 manage.py startproject tango

```
init.py:这是一个空的脚本,用来告诉Python编译器这个目录是一个Python包.
settings.py:用来存储Django项目设置的文件.
urls.py:用来存储项目里的URL模式.
wsgi.py:用来帮助你运行开发服务,同时可以帮助部署你的生产环境.
在新的 Djangoapp 项目中的文件：

init.py，前面的功能一样.
models.py,一个存储你的应用中数据模型的地方 - 在这里描述数据的实体和关系.
tests.py,存储你应用的测试代码.
views.py,在这里处理用户请求和响应.
admin.py,在这里你可以向Django注册你的模型,它会为你创建Django的管理界面.
新建一个项目后要在 setting.py 中设置 INSTALLED_APPS 元组 中 加入新建app的名字， 这样django就可以识别le。

### 创建视图
在views.py 中， 加入：

```python
from django.shortcuts import render
def index(requests):
    return render(requests, 'index.html')
```
需要在templates中新建一个html文件为index。

### 分配url
在urls.py中，添加代码：

```python
from django.conf.urls import url
from django.contrib import admin
from tango.views import index # 导入views层中的index

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^index/', index) # 增加urls
]
```
### 运行服务
在termainl中：

```python
python3 manage.py runserver

```
