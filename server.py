# -*- conding:utf-8 -*-
# file:server.py
# 
import tkinter
import threading 
import socket

class ListenThread(threading.Thread):       #监听线程
    def __init__(self,edit,server):
        threading.Thread.__init__(self)
        self.edit = edit        #保存客串中的多行文本框
        self.server = server
    def run(self):                          #进入监听状态
        while 1:                            #使用while循环等待连接
            try:
                client, addr = self.server.accept()
                self.edit.insert(tkinter.END,'连接来自：%s:%d\n' % addr)
                data = client.recv(1024).decode()        #接收数据
                self.edit.insert(tkinter.END,'收到数据：%s \n' % data)
                client.send(str('I GOT:%s' % data).encode())    #发送数据
                client.close()          #关闭同客户端的连接
                self.edit.insert(tkinter.END,'关闭客户端\n')
            except:     #监听线程已进入监听状态，在控制线程中壮志凌云socket连接将导致异常
                self.edit.insert(tkinter.END,'关闭连接\n')
                break
            
class Control(threading.Thread):
    def __init__(self,edit):
        threading.Thread.__init__(self)
        self.edit = edit
        self.event = threading.Event()          #创建Event对象
        self.event.clear()                      #清除event标志

    def run(self):
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   #创建socket连接
        server.bind(('',1051))                  #绑定本机的1051端口
        server.listen(1)                        #开始监听
        self.edit.insert(tkinter.END,'正在等待连接\n')
        self.lt = ListenThread(self.edit,server)    #创建监听线程对象
        self.lt.setDaemon(True)
        self.lt.start()
        self.event.wait()               #进入等待状态’
        server.close()                  #关闭连接
    def stop(self):
        self.event.set()                #设置event标志

class Window:
    def __init__(self,root):
        self.root = root
        self.butlisten = tkinter.Button(root,text='开始监听',command=self.Listen)   #创建组件
        self.butlisten.place(x =20,y=15)
        self.butclose=tkinter.Button(root,text='停止监听',command=self.Close)
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

