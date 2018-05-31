from collections import namedtuple
from collections import OrderedDict
class StockTradeDays(object):
    def __init__(self, price_array, start_date, date_array=None):
        #私有价格序列
        self.__price_array = price_array
        #私有日期序列
        #self.__date_array = self._init_days(start_date, date_array)
        #