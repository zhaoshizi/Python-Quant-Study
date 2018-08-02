# -*- conding:utf-8 -*-
import tkinter
import threading
import socket
import easytrader
import json
import logging
import time
import re
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import os.path

user = easytrader.use('ht_client')

class ListenThread(threading.Thread):       #监听线程
    def __init__(self,edit,log,server,control):
        threading.Thread.__init__(self)
        self.server = server
        self.edit = edit
        self.log = log
        self.control = control

    def run(self):                          #进入监听状态
        while 1:                            #使用while循环等待连接
            try:
                client, addr = self.server.accept()
                self.edit.insert(tkinter.END,'已连接')
                data = client.recv(1024).decode('utf-8')        #接收数据
                self.log.info('source data: %s' % data)
                #self.edit.insert(tkinter.END,'收到数据：%s \n' % data)

                data_json = json.loads(data)
                self.log.info('data-json: %s' % data_json)
                if data_json['operation'] == 'sell':
                    #do something
                    pass
                elif data_json['operation'] == 'buy':
                    #do something
                    pass
                elif data_json['operation'] == 'balance':
                    pass
                elif data_json['operation'] == 'cancel':
                    pass
                elif data_json['operation'] ==  'position':
                    pass
                elif data_json['operation'] == 'trades':
                    pass
                elif data_json['operation'] == 'entrust':
                    pass
                elif data_json['operation'] == 'stop':
                    self.log.info('线程收到命令: %s' % data_json['operation'])
                    self.control.stop()
                    continue
                else: 
                    #error
                    self.log.error('operation error: %s' % data_json['operation'])
                    continue
                    
                client.send(str('I GOT:%s' % data).encode())    #发送数据
                client.close()          #关闭同客户端的连接
                self.edit.insert(tkinter.END,'关闭客户端\n')
            except:     #监听线程已进入监听状态，在控制线程中壮志凌云socket连接将导致异常
                self.edit.insert(tkinter.END,'关闭监听\n')
                self.log.info('关闭监听')
                break
        self.server.close()

class Control(threading.Thread):
    def __init__(self,edit):
        threading.Thread.__init__(self)
        self.event = threading.Event()          #创建Event对象
        self.event.clear()                      #清除event标志
        self.edit = edit
        self.__createlog()
        self.log.info('Control-Thread created.')
    
    def __createlog(self):
        #1.创建一个log
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        #2.设置log的名字（manualthread日期.log
        rq = time.strftime('%s%Y%m%d', ('ManualThread',time.localtime(time.time())))
        log_path = os.path.dirname(os.getcwd()) + '/Logs/'
        log_name = log_path + rq + '.log'
        #3.日志打印格式
        log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
        formatter = logging.Formatter(log_fmt)
        #4.创建TimedRotatingFileHandler对象
        #interval: 滚动周期，单位有when指定，
        # 比如：when=’D’,interval=1，表示每天产生一个日志文件
        # backupCount: 表示日志文件的保留个数
        log_file_handler = TimedRotatingFileHandler(filename=log_name, when="D", interval=1, backupCount=30)
        log_file_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO)
        #5.handler添加到log
        self.log.addHandler(log_file_handler)
        

    def run(self):
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   #创建socket连接
        server.bind(('',1051))                  #绑定本机的1051端口
        server.listen(1)                        #开始监听
        self.edit.insert(tkinter.END,'手动线程正在等待连接\n')
        self.lt = ListenThread(self.edit,self.log,server,self)    #创建监听线程对象
        self.lt.setDaemon(True)
        self.lt.start()
        self.log.info('ListenThread Created.')
        self.event.wait()               #进入等待状态’
        server.close()                  #关闭连接
    def stop(self):
        self.event.set()                #设置event标志

class Window:
    def __init__(self,root):
        self.root = root
        self.butlisten = tkinter.Button(root,text='开始手动',command=self.Listen)   #创建组件
        self.butlisten.place(x =20,y=15)
        self.butclose=tkinter.Button(root,text='停止手动',command=self.Close)
        self.butclose.place(x=120,y=15)
        self.edit = tkinter.Text(root)
        self.edit.place(y=50)

    def Listen(self):
        self.ctrl = Control(self.edit)          #处理按钮事件
        self.ctrl.setDaemon(True)
        self.ctrl.start()

    def Close(self):
        self.ctrl.stop()

root = tkinter.Tk()
window = Window(root)
root.mainloop()

