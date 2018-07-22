import threading
import time


#设置一个全局变量，但这个全局变量是存放各线程的局部变量，各线程访问互不影响
selflocal = threading.local()
class ThreadCrawler(threading.Thread):
    def __init__(self,i):
        threading.Thread.__init__(self)
        self.sleeptime = i
    def run(self):
        selflocal.cnt =0
        while True:
            selflocal.cnt += 1
            print(selflocal.cnt,id(selflocal.cnt))
            time.sleep(self.sleeptime)

if __name__ == "__main__":
    a=1
    print(type(a))
    print(id(a))
    c= int(0)
    print(type(c))
    print(id(c))
    a=a+1
    print(id(a))
    c = c+1
    print(id(c))
    b=1
    print(id(b))
    b=b+1
    print(id(b))
    b=b+1
    print(a,b)
    print(id(b))
    tt = []
    for i in range(2):
        t = ThreadCrawler(i+1)
        t.setDaemon(True)
        t.start()
        tt.append(t)
    
    for i in tt:
        i.join()
