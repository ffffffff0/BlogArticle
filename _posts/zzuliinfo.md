---
title: 关于学校信息门户
date: 2019-5-23
tags: [Python, Boring, Bug]
---

学校信息门户的一些有趣的东西。

<!-- more -->
## 个人照片
信息门户登陆之后，发现个人照片的URL如下：
```
http://xxxxx/authentication/xxxx/getHeadImg?memberId=541510000000&pwd={SSHA}lhx1x5rxH9xP4c60xgEMaxSxUv1QM8xG6RXxVQ==

```
其中学号明文暴露，密码加密，更换学号可查看另一个人的头像。

*迭代可得--->2.2G*

## 个人信息
在信息门户中新增了一项功能，你可以寻找到你的老乡好友，本是好意，但是在这个功能中你可以好友的个人信息，如毕业院校，电话，籍贯地址，专业等，请求url为：

```
http://xxxxx/dmm_personal/xxx/xxx/paisanStu

```
只需要将学号作为data，向url发起post请求，返回数据就是此人的个人信息。

*迭代可得--->allinfo*

学校确实不太关注学生信息的安全，我在教务处上都看到学校把学生身份证号的信息公开的披露....................