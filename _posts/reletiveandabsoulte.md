---
title: CSS的定位方式
date: 2017-10-30
tags: [CSS]
categories: CSS
---
## 相对定位
![Alt text](	http://image-1252432001.coscd.myqcloud.com/FireShot%20Capture%201%20-%20Learn%20css%20with%20blocks%20-%20file____home_wen_python_Pyweb_blocks.html.png)

flower的红点就是相对定位的座标点(其实这个点是没有体积的，很像平面的概念)，我将flower移到第二层的位置。
<!-- more -->
```css
    .flower {
    box-sizing: border-box;
    width: 64px;
    height: 64px;
    background: url('images/rose.png');
    background-size: contain;
    // add
    position: relative;
    left: 64px;
    top: 64px;
}

```
改写flower的css，增加position属性，然后定位花移动的位置，移动了点，就像当于移动了flower。
![Alt text](http://image-1252432001.coscd.myqcloud.com/2.png)

## 绝对定位
绝对定位根据包含它父级元素作为座标系,且它的父级元素定位方式必须是相对定位或者绝对定位。
![Alt text](http://image-1252432001.coscd.myqcloud.com/3.png)

```css
    .yellow-flower {
  box-sizing: border-box;
  width: 64px;
  height: 64px;
  background: url('images/flower.png');
  background-size: contain;
  position: absolute;
  left: 128px;
}
```
当将yellow flower 移到第三层时，如果它的父级元素没有定位方式采用reletive or absolute，你发现它的红点和flower并不一样。

```html
<body>
  <div class="bg">
    <div class="flower">
      <div class="point">
      </div>
    </div>
    <div class="block-1"></div>
    <div class="yellow-flower">
      <div class="point">

      </div>
    </div>

    <div class="block-2"></div>

    <div class="block-3"></div>
  </div>

</body>
```
这里的yellow flower是根据body来移动的，然而这里body其实是向右偏移了8px(**CSS历史遗留问题**)，这个红点也是8px。
调整一下bg的定位方式即可或者可以修改body使body的margin为0。
![Alt text](http://image-1252432001.coscd.myqcloud.com/4.png)
## 居中方式
![Alt text](http://image-1252432001.coscd.myqcloud.com/5.png)

将图片居中

```css
.bg {
  border: solid 8px yellow;
  width: 320px;
  height: 256px;
  background-color: blue;
  position: absolute;
  left: 50%;
  top: 50%;
}
```
![Alt text](http://image-1252432001.coscd.myqcloud.com/6.png)

并没有对齐，但是其实又是对齐的，因为这里的重心在这幅画的上面的最左边的点，
这时需要改变重心的位置，这里使用transform。

```css
transform: translate(-50%,-50%);
```
![Alt text](http://image-1252432001.coscd.myqcloud.com/7.png)

**CSS的坑还是很多的。**
