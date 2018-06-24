import timeit 
#normal_list=range(10000)
#print(list(i**2 for i in normal_list))
#timeit1= timeit.Timer(stmt= 'list(i**2 for i in normal_list)',setup = 'normal_list=range(10000)')
#print(timeit1.timeit(number=10))

# print(timeit.timeit(stmt= 'list(i**2 for i in normal_list)',setup = 'normal_list=range(10000)',number=10))
# print(timeit.repeat(stmt= 'list(i**2 for i in normal_list)', setup='normal_list=range(10000)',repeat=2,number=10))
# print(timeit.timeit(stmt= 'list(i**2 for i in normal_list)',setup = 'a=10000;normal_list=range(a)',number=10))
# print(timeit.repeat(stmt= 'list(i**2 for i in normal_list)', setup='a=10000;normal_list=range(a)',repeat=2,number=10))

# def func():
#     normal_list=range(10000)
#     L = [i**2 for i in normal_list]

# print(timeit.timeit("func()", setup="from __main__ import func",number=10))
# print(timeit.repeat("func()", setup="from __main__ import func",repeat=2,number=10))

timer1 = timeit.Timer(stmt= 'list(i**2 for i in normal_list)',setup = 'normal_list=range(10000)')
print(timer1.timeit(number=10))
print(timer1.repeat(repeat=2,number=10))

timer1 = timeit.Timer(stmt= 'list(i**2 for i in normal_list)',setup = 'a=10000;normal_list=range(a)')
print(timer1.timeit(number=10))
print(timer1.repeat(repeat=2,number=10))

def func():
    normal_list=range(10000)
    L = [i**2 for i in normal_list]

timer1 = timeit.Timer("func()", setup="from __main__ import func")
print(timer1.timeit(number=10))
print(timer1.repeat(repeat=2,number=10))

import timeit 
import numpy as np
print(timeit.timeit(stmt='np_list = np.arange(10000);np_list**2',setup='import numpy as np',number = 100))
print(timeit.repeat(stmt='np_list = np.arange(10000);np_list**2',setup='import numpy as np',number = 100,repeat = 2))

def funcNp():
    np_list = np.arange(10000)
    np_list**2

print(timeit.timeit(stmt='funcNp()',setup='from __main__ import funcNp;import numpy as np',number = 100))
print(timeit.repeat(stmt='funcNp()',setup='from __main__ import funcNp;import numpy as np',number = 100,repeat = 2))

import numpy as np 
#200个股票
stock_cnt = 200
#504个交易日
view_days = 504
#生成服从正态分布：均值期望=0，标准差=1的序列
stock_day_change = np.random.standard_normal((stock_cnt,view_days))
#打印shape
print(stock_day_change.shape)
#打印出第一只股票，前5个交易日的涨跌幅情况
print(stock_day_change[0:1,:5])
#倒数一、二只股票，最后5个交易日数据
print(stock_day_change[-2:,-5:])

#Numpy内部实现的机制都是引用操作，用copy()是深拷贝
tmp = stock_day_change[0:2,0:5].copy()
#转换为int型
print(stock_day_change[0:2,0:5].astype(int))
#保留两位小数 np.around
print(np.around(stock_day_change[0:2,0:5],2))
#np.nan代表缺失
tmp_test = stock_day_change[0:2,0:5].copy()
#将第一个元素改成nan
tmp_test[0][0] = np.nan
print(tmp_test)
#np.where()语法句式类似java的三目运算符
tmp_test = stock_day_change[-2:,-5:]
print(tmp_test)
print(np.where(tmp_test>0.5,1,0))
print(np.where((tmp_test>0.5) | (tmp_test<-1),1,0))
#本地化保存和读取，保存时不加后缀，读取时加后缀
np.save('./gen/stock_day_change',stock_day_change)
stock_day_change = np.load('./gen/stock_day_change.npy')

#----------------------------------------
import numpy as np 
#200个股票
stock_cnt = 200
#504个交易日
view_days = 504
#生成服从正态分布：均值期望=0，标准差=1的序列
stock_day_change = np.random.standard_normal((stock_cnt,view_days))

stock_day_change_four = stock_day_change[:4,:4]
print(stock_day_change)
print('最大涨幅：{}'.format(np.max(stock_day_change_four,axis = 1)))
print('最小涨幅：{}'.format(np.min(stock_day_change_four,axis = 1)))
print('振幅幅度：{}'.format(np.std(stock_day_change_four,axis = 1)))
print('平均涨跌：{}'.format(np.mean(stock_day_change_four,axis = 1)))

#np.argmax()实现统计出最大的下标，下标从0开始，同理np.argmin()
import matplotlib
#a交易者，期望100，标准差50，
a_inverstor = np.random.normal(loc = 100,scale = 50,size=(100,1))
#a交易者期望
a_mean = a_inverstor.mean()
#a交易都标准差
a_std = a_inverstor.std()


