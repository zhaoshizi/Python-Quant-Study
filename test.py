price_str = '30.14, 29.58, 26.36, 32.56, 32.82'
print (type(price_str))

#id()函数获得对象的内存地址
print('旧的price_str id={}'.format(id(price_str)))
price_str = price_str.replace(' ','')
print('新的price_str id={}'.format(id(price_str)))
print(price_str)

price_array = price_str.split(',')
print(price_array)
#后面追加一个元素
price_array.append('32.82')
print(price_array)
#集合是一个无序容器，元素无重复
print(set(price_array))
price_array.remove('32.82')
print(price_array)

date_array = []
date_base = 20170118
#这里用for只是为了计数，无用的变量Python建议使用'_'声明
for _ in range(0,len(price_array)):
    date_array.append(str(date_base))
    #不考虑日期进位
    date_base += 1
print(date_array)

#使用while循环实现
date_array = []
date_base = 20170118
price_cnt = len(price_array)-1
while price_cnt >= 0:
    date_array.append(str(date_base))
    date_base += 1
    price_cnt -= 1
print(date_array)
#列表推导式
date_base = 20170118
date_array = [str(date_base + ind) for ind, _ in enumerate(price_array)]
print(date_array)

#tuple对象与字符串对象都是不可变对象
#zip的效果是同时迭代多个序列，每次分别从一个序列中取一个元素，一旦其中某个序列到达结尾，刚迭代宣告结束
stock_tuple_list = [(date, price) for date, price in zip(date_array, price_array)]
#tuple访问使用索引
print('20170119日价格：{}'.format(stock_tuple_list[1][1]))
print(stock_tuple_list)

#可命名元组namedtuple
from collections import namedtuple
stock_namedtuple = namedtuple('stock',('date','price'))
stock_namedtuple_list = [stock_namedtuple(date,price) for date, price in zip(date_array, price_array)]
#namedtuple访问使用price
print('20170119日价格：{}'.format(stock_namedtuple_list[1].price))
print(stock_namedtuple_list)

#字典推导式
#字典推导式：{key:vaule for in}
stock_dict = {date:price for date, price in zip(date_array, price_array)}
print('20170119日价格：{}'.format(stock_dict['20170119']))
print(stock_dict)
#dict使用key-value存储，特点是根据key查询value，速度快，使用keys()和values()函数可以分别返回字典中的key列表与value列表
print(stock_dict.keys(),stock_dict.values())

#有序字典 ordereddict
from collections import OrderedDict
stock_dict = OrderedDict((date, price) for date, price in zip(date_array, price_array))
print(stock_dict.keys())
print(stock_dict)

from collections import OrderedDict
stock_dict = OrderedDict([('20170118', '30.14'), ('20170119', '29.58'), ('20170120', '26.36'), ('20170121', '32.56'), ('20170122', '32.82')])
print(stock_dict)

min(zip(stock_dict.values(), stock_dict.keys()))

#自定义函数
def find_second_max(dict_array):
    #对传入的dict sorted排序
    stock_prices_sorted = sorted(zip(dict_array.values(), dict_array.keys()))
    #第二大值的函数也就是倒数第二个
    return stock_prices_sorted[-2]

#系统函数callable()验证是否为一个可调用（call）的函数
if callable(find_second_max):
    print(find_second_max(stock_dict))

#lambda函数
find_second_max_lambda = lambda dict_array:sorted(zip(dict_array.values(), dict_array.keys()))[-2]
print(find_second_max_lambda(stock_dict))

#Python函数可以返回多个值，打包为tuple
def find_max_and_min(dict_array):
    #对传入的dict sorted排序
    stock_prices_sorted = sorted(zip(dict_array.values(), dict_array.keys()))
    return stock_prices_sorted[0],stock_prices_sorted[-1]

print(find_max_and_min(stock_dict))

#map()函数，接收两个参数，一个是函数，一个是序列，map()把传入的函数依次作用于序列的每个元素，并把结果作为新的序列返回
#filter()函数，接收两个参数，一个是函数，一个是传感器，filter把传入的函数依次作用于每个元素，根据返回值是true还是false决定是
#保留还是丢弃该元素，结果序列是所有返回值为true的子集
#reduce()函数，把一个函数作用在一个序列上，这个函数必须接收两个参数，其中reduce函数把结果继续和序列的下一个元素做累积计算，
#reduce函数只返回值结果，非序列

#从收盘价格，推导出每天的涨跌幅度
from collections import OrderedDict
from functools import reduce
from collections import namedtuple
stock_dict = OrderedDict([('20170118', '30.14'), ('20170119', '29.58'), ('20170120', '26.36'), ('20170121', '32.56'), ('20170122', '32.82')])
#将字符串的价格通过列表推导式显示转换为float类型
#由于stock_dict是OrderedDict，所以才可以直接使用stock_dict.values获取有序日期的收盘价格
price_float_array = [float(price_str) for price_str in stock_dict.values()]
#通过将时间平移形成两个错开的收盘价格序列，通过zip让打包为一个新的序列
pp_array = [(price1,price2) for price1, price2 in zip(price_float_array[:-1], price_float_array[1:])]
print(pp_array)

#外层使用map()函数针对pp_array的每一个元素执行操作，内层使用reduce()函数即两个相邻的价格，求出涨跌幅度，返回外层结果list
#round将float保留几位小数，以下保留3位
change_array = list(map(lambda pp: reduce(lambda a, b:round((b-a) / a, 3),pp), pp_array))
change_array.insert(0,0)
print(change_array)

#使用namedtuple重新构建数据结构
stock_namedtuple = namedtuple('stock',('date','price','change'))
#通过zip分别从date_array,price_array,change_array拿数据组成stock_namedtuple然后以date作为key组成OrderedDict
stock_dict = OrderedDict((date,stock_namedtuple(date,price,change)) for date, price, change in zip(date_array,price_array,change_array))
print(stock_dict)
#使用filter进行数据筛选，筛选出上涨的交易日
up_days = list(filter(lambda day: day.change > 0,stock_dict.values()))
print(up_days)

#通用的函数来计算上涨或下跌的交易日，或计算所有上涨或下跌的涨幅和数据
#want_up 默认为True，want_calc_sum默认为False
def filter_stock(stock_array_dict, want_up=True, want_calc_sum=False):
    if not isinstance(stock_array_dict, OrderedDict):
        #如果类型不符合，刚产生错误
        raise TypeError('stock_array_dict must be OrderedDict!')
    #Python中的三目表达式的写法
    filter_func = (lambda day: day.change>0) if want_up else (lambda day: day.change <0)
    #使用filter_func作为筛选函数
    want_days = list(filter(filter_func, stock_array_dict.values()))
    
    if not want_calc_sum:
        return want_days
    #需要计算涨跌幅和
    change_sum = 0.0
    for day in want_days:
        change_sum += day.change
    return change_sum

#全部使用默认参数
print('所有上涨的交易日：{}'.format(filter_stock(stock_dict)))
#want_up = False
print('所有下跌的交易日：{}'.format(filter_stock(stock_dict,want_up=False)))
#计算所有上涨的总和
print('所有上涨交易日的涨幅和：{}'.format(filter_stock(stock_dict,want_calc_sum=True))) 
#计算所有下跌的总和
print('所有上涨交易日的涨幅和：{}'.format(filter_stock(stock_dict,want_up=False,want_calc_sum=True)))

#可以使用偏函数，创建新的函数，调用时更简单
from functools import partial
#筛选上涨交易日
filter_stock_up_days = partial(filter_stock, want_up=True, want_calc_sum=False)
#筛选下跌交易日
filter_stock_down_days = partial(filter_stock,want_up=False,want_clac_sum=False)
#筛选计算上涨交易日涨幅和
filter_stock_up_sums = partial(filter_stock,want_up=True,want_calc_sum=True)
#筛选计算下跌交易日跌幅和
filter_stock_down_sums = partial(filter_stock,want_up=False,want_calc_sum=True)