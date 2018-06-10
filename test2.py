from collections import namedtuple
from collections import OrderedDict
from functools import reduce

class StockTradeDays(object):
    def __init__(self, price_array, start_date, date_array=None):
        #私有价格序列
        self.__price_array = price_array
        #私有日期序列
        self.__date_array = self._init_days(start_date, date_array)
        #私有涨跌幅序列
        self.__change_array = self.__init_change()
        #进行OraderedDict的组装
        self.stock_dict = self._init_stock_dick()

    def __init_change(self):
        """
        从price_array生成change_array
        ：return：
        """
        price_float_array = [float(price_str) for price_str in self.__price_array]
        #通过将时间平移形成两个错开的收盘价序列，能过zip()函数打包成为一个新的序列
        #每个元素为相邻的两个收盘价格
        pp_array = [(price1,price2) for price1,price2 in zip(price_float_array[:-1],price_float_array[1:])]
        #涨跌幅的序列，map是函数作用于每一个序列元素，
change        _array = (list)map(lambda pp: reduce(lambda a, b :round((b-a)/a,3),pp),pp_array)
        #list_insert()函数插入数据，将第一天的涨跌幅设置为0
        change_array.insert(0,0)
        return change_array

    