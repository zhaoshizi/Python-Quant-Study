# -*- conding:utf-8 -*-
from kivy.config import Config
#设置中文字体
Config.set('kivy', 'default_font', [
    'msgothic',
    'fonts/msyh.ttc'])
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.listview import ListItemButton, ListItemLabel, \
        CompositeListItem, ListView
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.button import Button
import json
import AES
import base64
import socket

#data = [{'text':'a'}]
#print(data[0]['text'])
class ClientOperation(BoxLayout):
    def args_converter(self,row_index, rec):
        item_dict = { 'text': 'message',
            'size_hint_y': None,
            'height': 25}
        #动态添加列表的记录
        item_cls_dicts = [{'cls': ListItemButton,
                           'kwargs': {'text': str(i),'size_hint_x': None,'width': 250}} for i in rec.values()]
        item_dict['cls_dicts'] = item_cls_dicts
        return item_dict
        # return   { 'text': rec['text'],
        #     'size_hint_y': None,
        #     'height': 25,
        #     'cls_dicts': [{'cls': ListItemButton,
        #                    'kwargs': {'text': rec['text']}},
        #                    {
        #                        'cls': ListItemButton,
        #                        'kwargs': {'text': rec['text']}},
        #                    {
        #                        'cls': ListItemButton,
        #                        'kwargs': {'text': rec['text']}}]}

    def __init__(self,**kwargs):
        super(ClientOperation, self).__init__(**kwargs)
        self.mc = AES.MyCrypt('1234123412341234')
        #self.ids.listview_header_layout.bind(minimum_width=self.ids.listview_header_layout.setter('width'))
        #self.ids.listview_layout.bind(minimum_width=self.ids.listview_layout.setter('width'))
        self.ids.ScrollView_layout.bind(minimum_width=self.ids.ScrollView_layout.setter('width'))
        
        #self.ids.listview.bind(minimum_width=self.ids.listview.setter('width'))

    def show_list(self,data):
        listview_header_widgets = [Button(text=str(i),
                                         size_hint_y=None,
                                         size_hint_x=None,
                                         width =250,
                                         height=25) for i in data[0].keys()]
        #清空表头层，重新添加组件
        # self.ids.listview_header_layout.clear_widgets()
        # for i in listview_header_widgets:
        #     self.ids.listview_header_layout.add_widget(i)
        # self.ids.listview_header_layout.width = 250*len(data[0].keys())

        #设置列表的数据
        #self.ids.listview.adapter.data = data   
        #self.ids.listview.width = 250*len(data[0].keys())
        #self.ids.listview._trigger_reset_populate()
        #self.ids.ScrollView_layout.width = 250*len(data[0].keys())

    def show_text(self,data):
        self.ids.output.text = ''
        for i in data:
            for j,k in i.items():
                self.ids.output.text += str(j) + ': ' + str(k) +'\n'
    def send(self):
        message_dict = dict()
        message_dict['operation'] = self.ids.operation.text
        message_dict['code'] = self.ids.code.text
        message_dict['name'] = self.ids.name.text
        message_dict['price'] = self.ids.price.text
        message_dict['volume'] = self.ids.volume.text

        message_json = json.dumps(message_dict)
        print(message_json)
        encrypt_data = self.encrypt(message_json)

        try:
            ip = '127.0.0.1'
            port= 9001
            client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            client.connect((ip,port))
            client.send(encrypt_data)
            rdata=client.recv(20480)
            decrypt_data = self.decrypt(rdata)
            client.close()

            recieve_message = json.loads(decrypt_data)
            print(recieve_message)

            if message_dict['operation'] in ["sell", "buy", "balance", "cancel","position","trades","entrust"] :
                self.show_list(recieve_message)
            self.show_text(recieve_message)
        except Exception as e:
            self.ids.output.text = '发送错误\n' + repr(e)
        
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

class ClientApp(App):
    def build(self):
        return ClientOperation()

if __name__ == '__main__':
    ClientApp().run()
