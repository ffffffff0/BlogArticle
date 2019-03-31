---
title: 线程池实现爬虫
date: 2016-7-12
tags: [Python, Spider]
---
我将会写一个很无聊的爬虫，从一个url出发解析页面，提取页面中的链接，然后重复这个过程。

线程池方式：开一个线程池，每当爬虫发现一个新链接，就将链接放入任务队列中，线程池中的线程从任务队列获取一个链接，完成抓取页面、解析、将新连接放入工作队列。
<!-- more -->
## 创建线程
通过继承线程类，来创建线程，
线程同时操作一个全局变量时会产生线程竞争所以需要锁，直到锁被释放时，才能操纵更改，任何时候只能存在一个线程存在一个锁。
利用queue来进行线程之间的通信。

```python
class Fetcher(Thread):
    # 继承thread类
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        # 设置守护线程
        self.daemon = True

        self.start()

    def run(self):
        while True:
            # 从队列去除一个item， queue为空时则阻塞
            url = self.tasks.get()
            print(url)
            links = self.parse_links(url)
            lock.acquire()
            # 获得锁
            # 将得到新连接放到任务队列与seen_urls
            for link in links.difference(seen_urls):
                # 迭代不同于seen_urls的链接
                self.tasks.put(link)
                # 放进队列，queue满时则阻塞
            seen_urls.update(links)
            # 释放锁
            lock.release()
            # 通知任务队列这个线程的任务完成
            self.tasks.task_done()
# 解析页面
    def parse_links(self, fecher_url):
        try:
            soup = requests.get(fecher_url)
            urls = set(re.findall(r'''(?i)href=["']?([^\s"'<>]+)''', soup.text))
            links = set()
            for url in urls:
                if url.split(":")[0] not in ('http', 'https'):
                    continue
                else:
                    links.add(url)
            return links
        except:
            return set()
```
### 实现线程池

```python
class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = Queue()
        for _ in range(num_threads):
            Fetcher(self.tasks)

    def add_task(self, url):
        self.tasks.put(url)
        # 将url放进队列

    def wait_completion(self):
        self.tasks.join()
        # 阻塞
```
### main部分运行程序

```python

if __name__ == '__main__':
    start = time.time()
    pool = ThreadPool(4)
    pool.add_task('http://zzuli.edu.cn/')
    pool.wait_completion()
    print('{} urls fetcher in {:.1f} seconds'.format(len(seen_urls), time.time() - start))

```
