---
title: GOT 词云图
date: 2016-5-4
tags: [Python, 权力的游戏]
---

分析GOT文本， 利用jieba中文分词库和wordcloud来绘制词云图。
<!-- more -->
```python
import jieba.analyse

from wordcloud import WordCloud

from PIL import Image

import codecs

import matplotlib.pyplot as plt

import numpy as np
from collections import Counter

get_ipython().magic('matplotlib')

f = codecs.open('/home/jasd/python/game_of_throne.txt', 'r', 'utf-8')
content = f.read()
f.close()

mask = np.array(Image.open('/home/jasd/python/ml/saber.jpg'))

word = jieba.cut(content, cut_all=True)
wordpro = ''.join(word)

wordcloud = WordCloud(font_path='./fonts/wqy-microhei.ttc', margin=5, mask=mask, max_words=1000)
wordcloud = wordcloud.generate(wordpro)

plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# 关键词权重

words = jieba.analyse.extract_tags(content, topK=100, withWeight=True)

for word, freq in words:
    print(word, freq)

```

如图；
![image](https://ooo.0o0.ooo/2017/05/07/590efdc75f2c5.png)
