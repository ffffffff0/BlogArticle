---
title: Python和图书馆网站
date: 2016-5-5
tags: [Python, Library]
categories: [Python, spider]
---

前段时间看完了python网络数据收集这本书，自己想练习一下。
联想到刚入学时的时候登陆学校图书馆网站，初始密码是学号，觉得很鸡肋，
觉得用模拟登陆和验证码识别，就可以爬下我们学院每个人的图书借阅记录和个人信息，顺便写一个自动借阅的脚本。
顿时很有兴趣，就动手了。
<!-- more -->
- 验证码
验证码很顺利，学校的图书馆的验证码很容易识别，就用pytesseract很容易。

```python
import pytesseract
from PIL import Image
import requests
captcha_url ="http://202.196.13.8:8080/reader/captcha.php"
img = requests.get(captcha_url).content
code = pytesseract.image_to_string(img)
print(code)

```
- Cookie
其实cookie的处理，我也学习一些库的使用，起初实验了很多次，都是无法进入登陆后的页面，
后来用了一个库http.cookiejar就顺利解决了。

```python
# 打开session。
session = requests.Session()
# 保存cookies
session.cookies = cookielib.LWPCookieJar(filename='cookies')
try:
    # session加载cookies
    session.cookies.load(ignore_discard=True)
except:
    print("Cookie 未能加载")

```
- 编码
编码真是蛋疼到爆呀，每次print都是一堆乱码，编码让我怀疑人生的说，
我也知道cmd上print不了UTF-8，我的atom也是如此，pycharm也要改一下encoding，
这是才知python的idle才是最好用的。

```python
info = session.get("http://202.196.13.8:8080/reader/redr_info.php",
                   headers=headers).content.decode("UTF-8")
```
### code:

```python

import requests
from bs4 import BeautifulSoup
import time
import http.cookiejar as cookielib
import pymongo
import pytesseract
from PIL import Image


# 连接mongodb
client = pymongo.MongoClient("localhost", 27017, connect=False)
library = client['library']
library_info = library['ys13']
# 设置headers
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'zh-CN,zh;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Host': '202.1X6.13.X:8080',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36'
}
# 打开session。
session = requests.Session()
# 保存cookies
session.cookies = cookielib.LWPCookieJar(filename='cookies')
try:
    # session加载cookies
    session.cookies.load(ignore_discard=True)
except:
    print("Cookie 未能加载")



def get_captcha():
    captcha_url = "http://202.196.13.8:8080/reader/captcha.php"
    x = 0
    bin = session.get(captcha_url, headers=headers).content
    with open("/home/jasd/python/%s.jpg" % x, "wb")as file:
        file.write(bin)
    image = Image.open("/home/jasd/python/0.jpg")
    code = pytesseract.image_to_string(image)
    return code


# get_chaptcha()主要使用tesseract来通过subprocess来将验证码识别
# login()为登陆函数，可采集集信息， 并放置于mongodb中


def login(xuehao):
    from_data = {
        'number': xuehao,
        'passwd': xuehao,
        'captcha': get_captcha(),
        'select': 'cert_no',
        'returnUrl': '',
        }
    session.post("http://202.196.13.8:8080/reader/redr_verify.php",
                 data=from_data, headers=headers)
    info = session.get("http://202.196.13.8:8080/reader/redr_info.php",
                       headers=headers).content.decode("UTF-8")
    soup = BeautifulSoup(info, "lxml")
    if soup.find('div', {'id': 'mylib_content'}):
        mylib_info = soup.findAll("td")
        mylib_msg = soup.findAll("a", {'href': "book_lst.php"})
        data = {
            '姓名': mylib_info[1].get_text().split("：")[1],
            '证件号': mylib_info[2].get_text().split("：")[1],
            '累计借书': mylib_info[12].get_text().split("：")[1],
            '五天内即将过期图书': mylib_msg[1].get_text().split("[")[1].split("]")[0],
            '已超期图书': mylib_msg[2].get_text().split("[")[1].split("]")[0],
            '欠款金额': mylib_info[14].get_text().split("：")[1],
            '工作单位': mylib_info[18].get_text().split("：")[1],
            '职业/职称': mylib_info[19].get_text().split("：")[1],
            '性别': mylib_info[21].get_text().split("：")[1],
            '出生日期': mylib_info[26].get_text().split("：")[1],
            }
        print(data)
        library_info.insert_one(data)
        session.cookies.save()
    else:
        pass


if __name__ == '__main__':
    time.sleep(2)
    for i in range(541310020101, 541310020161):
        login(str(i))

```

### 通过分析数据可以得到以下图表：

在jupyter中简单的分析：

```python
# coding: utf-8
import pymongo
import charts
from functools import reduce


client = pymongo.MongoClient('localhost', 27017)
scraping = client['scraping']
library = client['library']
IdNumbers = scraping['IdNumbers']
xk15 = library['xk15']
ys15 = library['ys15']
math_all = library['math_all']
iec = library['iec_all']

all_loc = []
for ma in iec.find():
    for d in IdNumbers.find():
        if reduce(lambda x, y:  int(str(x) + str(y)), list(str(ma['身份证号']))[0: 7]) == int(d['number']):
          all_loc.append(d['city'])
#           print(d['city'], ma['姓名'])

locs = [reduce(lambda x, y: str(x) + str(y), list(loc)[0: 2]) for loc in all_loc]
list(set(locs))

loc_times = []
for i in list(set(locs)):
    loc_times.append(locs.count(i))
print(loc_times)

def gen_data(types):
    length = 0
    if length <= len(locs):
        for area, times in zip(list(set(locs)), loc_times):
            data = {
                'name': area,
                'data': [times],
                'type': types,
            }
            yield data
            length += 1

for i in gen_data('column'):
    print(i)

series = [data for data in gen_data('column')]
charts.plot(series, show='inline', options=dict(title=dict(text='学院学生分布')))

def gen_datas():
    for area, times in zip(list(set(locs)), loc_times):
        yield [area, times]

for i in gen_datas():
    print(i)

options = {
    'chart'   : {'zoomType':'xy'},
    'title'   : {'text': ' 数学学院'},
    'subtitle': {'text': '学生分布'},
    }
series =  [{
    'type': 'pie',
    'name': 'pie charts',
    'data':[i for i in gen_datas()]

        }]
charts.plot(series,options=options,show='inline')

```
![imgage](https://image-1252432001.cos.ap-chengdu.myqcloud.com/zzuliinfo/chart_math.png)


其实我的目的是写一个能够知道我借阅的图书到期是给我发mail，提醒我，如果可以续借帮我续借的脚本:[Renewbook](https://github.com/jianaosiding/spider_code/blob/master/renew.py)
