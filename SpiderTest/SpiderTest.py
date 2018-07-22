# -*- coding: utf-8 -*-

from queue import Queue
import threading
import urllib
from bs4 import BeautifulSoup
import random
import codecs
import json
import time

class ThreadCrawler(threading.Thread):
    def __init__(self,url_queue, data_queue, url_set, lock, hds=""):
        threading.Thread.__init__(self)
        self.url_set = url_set
        self.url_queue = url_queue
        self.data_queue = data_queue
        self.lock = lock
        self.hds = hds
        self.request_url = u'https://bj.lianjia.com/ershoufang/esfrecommend?id='
        self.localcnt =0

    def run(self):
        print('{} strat'.format(self.name))
        self._CrawlMsg()

    def _CrawlMsg(self):
        global cnt
        while True:
            url = self.url_queue.get()
            if url == "":
                return
            try:
                #获取推荐的链接
                #获取链家编号，其实下面从页面也能获取
                index1 = url.rindex('/')
                index2 = url.index('.html')
                url_id = url[index1 + 1:index2]
                hds = self.hds[random.randint(0, len(self.hds) - 1)]
                #发送请求，获取推荐的数据
                req = urllib.request.Request(
                    self.request_url + url_id, headers=hds)
                recommend_code = urllib.request.urlopen(req).read()
                recommend_json = json.loads(recommend_code.decode('utf-8'))
                #从json格式的报文读取推荐的链接
                for i in range(0, len(recommend_json['data']['recommend'])):
                    recommend_url = recommend_json['data']['recommend'][i][
                        'viewUrl']
                    with self.lock:
                        if recommend_url not in self.url_set:
                            self.url_set.add(recommend_url)
                            self.url_queue.put(recommend_url)
                        else:
                            continue
                #获取当前编号的页面，进行解析
                req = urllib.request.Request(url, headers=hds)
                source_code = urllib.request.urlopen(req).read()
                source_code = source_code.decode('utf-8')
                page_struct = BeautifulSoup(source_code, "lxml")

                #总价
                DataString = page_struct.find('span', {
                    'class': 'total'
                }).string + ','
                #单价
                DataString = DataString + page_struct.find(
                    'span', {
                        'class': 'unitPriceValue'
                    }).next + ','
                #房屋基本信息
                for i in page_struct.find('div', {
                        'class': 'houseInfo'
                }).stripped_strings:
                    DataString = DataString + i + ','
                #房屋的详细信息
                #找到('span', {'class':'label'})的下一个兄弟节点，只要第一条文字信息
                for i in page_struct.findAll('span', {'class': 'label'}):
                    #特殊处理
                    if i.string == "链家编号":
                        for j in i.next_sibling.stripped_strings:
                            DataString = DataString + j + ','
                            break
                    #特殊处理
                    elif i.string == "所在区域":
                        for j in i.next_sibling.stripped_strings:
                            DataString = DataString + j + ','
                    else:
                        if i.next_sibling.string != '\n':
                            DataString = DataString + i.next_sibling.string + ','
                        else:
                            for j in i.next_sibling.next_sibling.stripped_strings:
                                DataString = DataString + j + ','
                                break
                DataString = DataString + url
                self.data_queue.put(DataString)
                #print(DataString)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(repr(e),'\n',url)
                continue
            except (AttributeError,TypeError) as e:
                print(repr(e),'\n',url)
                if 'NoneType' in str(e):
                    self.url_queue.put(url)
                    print('put url into url_queue')
                continue
            except (ConnectionResetError) as e:
                print(repr(e),'\n',url)
                if '10054' in str(e):
                    self.url_queue.put(url)
                    print('put url into url_queue')
                continue
            except Exception as e:
                print(repr(e),'\n',url)
                continue
            finally:
                self.url_queue.task_done()
            self.localcnt += 1
            if self.localcnt % 10 == 0:
                print("{}正常，已获取{}条记录".format(self.name,self.localcnt))
                time.sleep(2)
            with self.lock:
                #cnt声明在main中了，虽然提示有错，但不影响执行，如果声明在最外面这里就不报错
                cnt = cnt + 1
                if cnt % 50 == 0:
                    print("共获取{}条数据，由{}输出".format(cnt,self.name))
            


class ThreadWriter(threading.Thread):
    def __init__(self, data_queue, lock, data_file):
        threading.Thread.__init__(self)
        self.data_queue = data_queue
        self.lock = lock
        self.data_file = data_file

    def run(self):
        cnt = 0
        while True:
            data_str = self.data_queue.get() + '\n'
            with self.lock:
                data_file.write(data_str)
                cnt = cnt + 1
            self.data_queue.task_done()
            if cnt % 10 == 0:
                data_file.flush()


if __name__ == "__main__":
    #Some User Agents
    hds=[{'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'},\
        {'User-Agent':'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'},\
        {'User-Agent':'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'},\
        {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0'},\
        {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/44.0.2403.89 Chrome/44.0.2403.89 Safari/537.36'},\
        {'User-Agent':'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
        {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
        {'User-Agent':'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'},\
        {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
        {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
        {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11'},\
        {'User-Agent':'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11'},\
        {'User-Agent':'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11'}]
    global cnt
    cnt = 0
    urls_queue = Queue()
    data_queue = Queue()
    url_set = set()
    data_file = codecs.open('houseinfo.csv', 'w', 'utf8')
    lock = threading.Lock()
    src_url = u"https://bj.lianjia.com/ershoufang/101101825927.html"
    file_head = "总价,单价,房屋户型,所在楼层,房屋朝向,装修情况,建筑面积,建造年代,小区名称,城区,街道,环数,看房时间,链家编号," + \
                "房屋户型,所在楼层,建筑面积,户型结构,套内面积,建筑类型,房屋朝向,建筑结构,装修情况," + \
                "梯户比例,供暖方式,配备电梯,产权年限,挂牌时间,交易权属,上次交易,房屋用途,房屋年限,产权所属,抵押信息,房本备件,链接"
    url_set.add(src_url)
    urls_queue.put(src_url)
    data_queue.put(file_head)
    for i in range(4):
        t = ThreadCrawler(urls_queue, data_queue, url_set, lock, hds=hds)
        t.setDaemon(True)
        t.start()

    t = ThreadWriter(data_queue, lock, data_file)
    t.setDaemon(True)
    t.start()

    urls_queue.join()
    data_queue.join()
    with lock:
        data_file.close()