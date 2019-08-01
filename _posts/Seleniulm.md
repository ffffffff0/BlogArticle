---
title: selenium实现浏览器下拉页面
date: 2016-10-10
tags: [Python, Selenium, Spider]
categories: [Python, spider]
---

在我爬取美团数据的时候，发现美团的整个页面数据采用ajax来加载，ajax 就是异步的javascript和XML, 我也不是很懂， 近期打算学习js，它的功能就是网站不需要使用单独的页面请求就可以和服务器进行交互，表单采用Ajax与服务器通信。
<!-- more -->
要想获得全部的数据，需要下拉到底页面才会完全加载本页面的全部内容。

这里其时有两种办法，来进行爬取：

抓包分析请求，找到真实的请求，来模拟请求，获得数据。

使用selenium来进行爬取，但是其效率不高。

由于我爬取的数据量较小，我只需要几个分类下的数据。我采用第二种。（第一种，模拟起来很麻烦）。

```python
# 导入第三方库
from selenium import webdriver
import time
from bs4 import BeautifulSoup

# 你可以使用 phantomjs 或者使用 chrome
driver = webdriver.PhantomJS(executable_path='/home/jasd/python/phantomjs/bin/phantomjs')
# driver = webdriver.Chrome()

def get_info(page):

    driver.get(full_url)
    time.sleep(2)
# 这里我使用 js 的方法，来实现下拉页面
    js = "window.scrollTo(0,document.body.scrollHeight)"
# 使用的执行 JavaScript
    driver.execute_script(js)
# 休眠 3 秒种
    time.sleep(3)
# 重复执行多次， 确保页面已完全加载
    for i in range(4):
        js = "window.scrollTo(0,document.body.scrollHeight)"
        driver.execute_script(js)
        time.sleep(3)

# 这里使用 page_source 的方法， 我还是比较喜欢使用BeautifulSoup， 虽然效率有点低
    pageSoure = driver.page_source
    soup = BeautifulSoup(pageSoure, 'lxml')
# 退出
    driver.quit()

get_info(i)

```

我觉得代码写的很挫。。。。
