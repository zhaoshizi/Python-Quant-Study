import threading
import random, time
 
class ThreadLocal():
    def __init__(self):
        self.local = threading.local()
    def run(self):
        time.sleep(random.random())
        self.local.number =[]
        for i in range (10):
            self.local.number.append(random.choice(range(10)))
        print(threading.currentThread(), self.local.number)

a = 1
b = 1 
print(id(a))
print(id(b))
a = a+1
print(a)
print(b)
threadLocal = ThreadLocal()
threads = []
for i in range(5):
    t = threading.Thread(target=threadLocal.run)
    t.start()
    threads.append(t)
for i in range(5):
    threads[i].join