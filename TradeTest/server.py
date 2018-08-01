# -*- conding:utf-8 -*-

import threading
import socket
import easytrader

class ListenThread(threading.Thread):       #监听线程
    def __init__(self,user,server):
        threading.Thread.__init__(self)
        self.user = user        
        self.server = server
        self.flag = True
    def run(self):                          #进入监听状态
        while self.flag:                            #使用while循环等待连接
            try:
                client, addr = self.server.accept()
                #self.edit.insert(tkinter.END,'连接来自：%s:%d\n' % addr)
                data = client.recv(1024).decode()        #接收数据
                #self.edit.insert(tkinter.END,'收到数据：%s \n' % data)
                args = data.split('|')
                if args[0] == 'sell':
                    #do something
                    pass
                elif args[0] == 'buy':
                    #do something
                    pass
                elif args[0] == 'balance':
                    pass
                elif args[0] == 'cancel':
                    pass
                elif args[0] ==  'position':
                    pass
                elif args[0] == 'trades':
                    pass
                elif args[0] == 'entrust':
                    pass
                elif args[0] == 'stop':
                    self.flag = False
                    continue
                else: 
                    pass
                    #error
                client.send(str('I GOT:%s' % data).encode())    #发送数据
                client.close()          #关闭同客户端的连接
                #self.edit.insert(tkinter.END,'关闭客户端\n')
            except:     #监听线程已进入监听状态，在控制线程中壮志凌云socket连接将导致异常
                #self.edit.insert(tkinter.END,'关闭连接\n')
                break
        server.close()

class Control(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.event = threading.Event()          #创建Event对象
        self.event.clear()                      #清除event标志
        self.user = easytrader.use('ht_client')

    def run(self):
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   #创建socket连接
        server.bind(('',1051))                  #绑定本机的1051端口
        server.listen(1)                        #开始监听
        #self.edit.insert(tkinter.END,'正在等待连接\n')
        self.lt = ListenThread(self.user,server)    #创建监听线程对象
        self.lt.setDaemon(True)
        self.lt.start()
        self.event.wait()               #进入等待状态’
        server.close()                  #关闭连接
    def stop(self):
        self.event.set()                #设置event标志


