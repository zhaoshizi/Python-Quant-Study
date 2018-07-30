# _*_ coding:utf-8 _*_
# file:client.py
#
import tkinter
import socket
class Window:
    def __init__(self,root):
        label1 = tkinter.Label(root,text = 'IP')
        label2 = tkinter.Label(root,text='Port')
        label3 = tkinter.Label(root,text='Data')
        label1.place(x=5,y=5)
        label2.place(x=5,y=30)
        label3.place(x=5,y=55)
        self.entryIP = tkinter.Entry(root)      #输入框
        self.entryIP.insert(tkinter.END,'127.0.0.1')
        self.entryPort = tkinter.Entry(root)
        self.entryPort.insert(tkinter.END,'1051')
        self.entryData = tkinter.Entry(root)
        self.entryData.insert(tkinter.END,'Hello')
        self.Recv = tkinter.Text(root)
        self.entryIP.place(x=40,y=5)
        self.entryPort.place(x=40,y=30)
        self.entryData.place(x=40,y=55)
        self.Recv.place(y=115)
        self.send = tkinter.Button(root,text='发送数据',command=self.Send)
        self.send.place(x=40,y=80)

    def Send(self):
        try:
            ip = self.entryIP.get()
            port= int(self.entryPort.get())
            data=self.entryData.get()
            client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            client.connect((ip,port))
            client.send(data.encode())
            rdata=client.recv(1024)
            self.Recv.insert(tkinter.END,'Server:'+rdata.decode() + '\n')
            client.close()
        except:
            self.Recv.insert(tkinter.END,'发送错误\n')

root = tkinter.Tk()
window = Window(root)
root.mainloop()