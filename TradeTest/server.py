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
import AES,base64

user = easytrader.use('ht_client')

class ListenThread(threading.Thread):       #监听线程
    def __init__(self,edit,log,server,control):
        threading.Thread.__init__(self)
        self.server = server
        self.edit = edit
        self.log = log
        self.control = control
        self.mc = AES.MyCrypt('1234123412341234')

    def run(self):                          #进入监听状态
        while 1:                            #使用while循环等待连接
            try:
                client, addr = self.server.accept()
                self.edit.insert(tkinter.END,'已连接')
                data = client.recv(2048).decode('utf-8')        #接收数据
                self.log.info('source data: %s' % data)
                #self.edit.insert(tkinter.END,'收到数据：%s \n' % data)
                data = self.decrypt(data)

                data_json = json.loads(data)

                message_dict = dict()
                self.log.info('data-json: %s' % data_json)
                if data_json['operation'] == 'sell':
                    #do something
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'entrust_no': 'xxxxxxxx'}]
                    pass
                elif data_json['operation'] == 'buy':
                    #do something
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'entrust_no': 'xxxxxxxx'}]
                    pass
                elif data_json['operation'] == 'balance':
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'参考市值': 21642.0,
                                    '可用资金': 28494.21,
                                    '币种': '0',
                                    '总资产': 50136.21,
                                    '股份参考盈亏': -90.21,
                                    '资金余额': 28494.21,
                                    '资金帐号': 'xxx'}]
                    pass
                elif data_json['operation'] == 'cancel':
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'message': '撤单申报成功'}]
                    pass
                elif data_json['operation'] ==  'position':
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'买入冻结': 0,
                                    '交易市场': '沪A',
                                    '卖出冻结': '0',
                                    '参考市价': 4.71,
                                    '参考市值': 10362.0,
                                    '参考成本价': 4.672,
                                    '参考盈亏': 82.79,
                                    '当前持仓': 2200,
                                    '盈亏比例(%)': '0.81%',
                                    '股东代码': 'xxx',
                                    '股份余额': 2200,
                                    '股份可用': 2200,
                                    '证券代码': '601398',
                                    '证券名称': '工商银行'}]
                    pass
                elif data_json['operation'] == 'trades':
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'买卖标志': '买入',
                                    '交易市场': '深A',
                                    '委托序号': '12345',
                                    '成交价格': 0.626,
                                    '成交数量': 100,
                                    '成交日期': '20170313',
                                    '成交时间': '09:50:30',
                                    '成交金额': 62.60,
                                    '股东代码': 'xxx',
                                    '证券代码': '162411',
                                    '证券名称': '华宝油气'}]
                    pass
                elif data_json['operation'] == 'entrust':
                    self.log.info('operation: ' + data_json['operation'] )
                    message_dict = [{'买卖标志': '买入',
                                    '交易市场': '深A',
                                    '委托价格': 0.627,
                                    '委托序号': '111111',
                                    '委托数量': 100,
                                    '委托日期': '20170313',
                                    '委托时间': '09:50:30',
                                    '成交数量': 100,
                                    '撤单数量': 0,
                                    '状态说明': '已成',
                                    '股东代码': 'xxxxx',
                                    '证券代码': '162411',
                                    '证券名称': '华宝油气'},
                                    {'买卖标志': '买入',
                                    '交易市场': '深A',
                                    '委托价格': 0.6,
                                    '委托序号': '1111',
                                    '委托数量': 100,
                                    '委托日期': '20170313',
                                    '委托时间': '09:40:30',
                                    '成交数量': 0,
                                    '撤单数量': 100,
                                    '状态说明': '已撤',
                                    '股东代码': 'xxx',
                                    '证券代码': '162411',
                                    '证券名称': '华宝油气'}]
                    pass
                elif data_json['operation'] == 'stop':
                    self.log.info('线程收到命令: %s' % data_json['operation'])
                    message_dict = [{'message': 'linster由客户端关闭'}]
                    self.edit.insert(tkinter.END,'关闭监听，由客户端关闭\n')
                    self.log.info('linster由客户端关闭')
                    #self.control.stop()
                    break
                elif data_json['operation'] == 'login':
                    #user.prepare('/path/to/your/yh_client.json') # 配置文件路径
                    message_dict = [{'message': 'login'}]
                elif data_json['operation'] == 'logout':
                    #user.exit()
                    message_dict = [{'message': 'logout'}]
                elif data_json['operation'] == 'None':
                    message_dict = [{'message': 'do nothing'}]
                else: 
                    #error
                    self.log.error('operation error: %s' % data_json['operation'])
                    message_dict = [{'message': 'operation error'}]
                    continue
            except Exception as e:     #监听线程已进入监听状态，在控制线程中壮志凌云socket连接将导致异常
                message_dict = [{'message': 'Listener error ' + repr(e) }]
                self.edit.insert(tkinter.END,'关闭监听\n')
                self.log.info('message:' ' Listener error ' + repr(e))
                self.log.info('关闭监听')
                break
            finally:
                #转为json
                message_json = json.dumps(message_dict)
                encrypt_data = self.encrypt(message_json)
                #判断client存在不存在
                if 'client' in dir():
                    client.send(encrypt_data)    #发送数据
                    client.close()          #关闭同客户端的连接
                self.edit.insert(tkinter.END,'关闭客户端\n')

        #self.server.close()
        self.control.stop()

    def encrypt(self,message):
        #对数据加密
        encrypt_data = self.mc.encrypt(message)
        #使用base64转码
        return base64.b64encode(encrypt_data)
    def decrypt(self,message):
        #使用base64解码
        encrypt_data = base64.b64decode(message)
        #对数据解密
        return self.mc.decrypt(encrypt_data)

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
        #2.设置log的名字（manualthread日期.log）
        rq = time.strftime('%Y%m%d', time.localtime(time.time()))
        log_path = os.path.dirname(__file__) + '/Logs/'
        log_name = log_path + 'ManualThread' + rq + '.log'
        #Logs文件夹不存在创建Logs文件夹
        if not os.path.exists(os.path.dirname(__file__) + '/Logs/'):
            os.makedirs(os.path.dirname(__file__) + '/Logs/')
        #3.日志打印格式
        log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
        formatter = logging.Formatter(log_fmt)
        #4.创建TimedRotatingFileHandler对象
        #interval: 滚动周期，单位有when指定，
        # 比如：when=’D’,interval=1，表示每天产生一个日志文件
        # backupCount: 表示日志文件的保留个数
        log_file_handler = TimedRotatingFileHandler(filename=log_name, when="D", interval=1, backupCount=30,encoding='utf-8')
        log_file_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO)
        #5.handler添加到log
        self.log.addHandler(log_file_handler)
        

    def run(self):
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   #创建socket连接
        server.bind(('',9001))                  #绑定本机的9001端口
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

