import itertools
import test3
import test2
from functools import reduce

items = [1, 2, 3]
#排列
print('排列：')
for item in itertools.permutations(items):
    print(item)
#无放回组合
print('无放回组合：')
for item in itertools.combinations(items, 2):
    print(item)
#有放回组合
print('有放回组合：')
for item in itertools.combinations_with_replacement(items, 2):
    print(item)

#笛卡尔积，针对多个输入序列进行排列组合
ab = ['a', 'b']
cd = ['c', 'd']
#针对ab、cd两个集合进行排列组合
print('笛卡尔积：')
for item in itertools.product(ab, cd):
    print(item)

def calc(keep_stock_threshold,buy_change_threshold):
    """
    :param keep_stock_threshold：持股天数
    :param buy_change_threshold：下跌买入阀值
    :return:盈亏情况，输入的持股天数，输入的下跌买入阀值
    """
    #实例入TradeStrategy2
    trade_strategy2 = test3.TradeStrategy2()
    #通过类方法设置买入后持股天数
    test3.TradeStrategy2.set_keep_stock_threshold(keep_stock_threshold)
    #通过类方法设置下跌买入阀值
    test3.TradeStrategy2.set_buy_change_threshold(buy_change_threshold)
    #进行回测
    trade_loop_back = test3.TradeLoopBack(test2.trade_days,trade_strategy2)

    trade_loop_back.execute_trade()
    #计算回测结果的最终盈亏值profit
    profit = 0.0 if len(trade_loop_back.profit_array) == 0 else reduce(lambda a,b:a+b,trade_loop_back.profit_array)
    #返回值profit和函数的两个输入参数
    return profit,keep_stock_threshold,buy_change_threshold

#测试
calc(20,-0.08)

#笛卡尔各求最优属于有限参数范围内求最优的问题
#range集合：买入后持股天数从2~30天，间隔两天
keep_stock_list =list(range(2,30,2))
print('持股天数参数组：{}'.format(keep_stock_list))
#下跌买入阀值从-0.05到-0.15，即从5%下跌到15%
buy_change_list=[buy_change /100.0 for buy_change in range(-5,-16,-1)]
print('下跌阀值参数组:{}'.format(buy_change_list))

result = []
for keep_stock_threshold,buy_change_threshold in itertools.product(keep_stock_list,buy_change_list):
    #使用calc()函数计算参数对应的最终盈利，结果加入result序列
    result.append(calc(keep_stock_threshold,buy_change_threshold))
print('笛卡尔积参数集合总共结果为：{}个'.format(len(result)))

#[::-1]疳整个排序结果反转，反转后盈亏收益从最高向低开始排序。其中[::-1]代表从后向前取值，每次步进值为1
#[:-10]取出收益最高的前10个组合查看
print(sorted(result)[::-1][:10])

#多进程与多线程
#由于全局解释器GIL，Python的线程被限制为同一时刻只允许一个线程执行，
#所以Python的多线程适用于处理I/O密集任务和并发执行的阻塞操作，多进程处理并行的计算密集型任务

#使用多进程
from concurrent.futures import ProcessPoolExecutor
result =[]
#回调函数，通过add_done_callback任务完成后调用
def when_done(r):
    #when_done在主进程中运行
    result.append(r.result())
"""
    with class_a() as a:上下文管理器
"""
def main():
    with ProcessPoolExecutor() as pool:
        for keep_stock_threshold, buy_change_threshold in itertools.product(keep_stock_list,buy_change_list):
            """
                submit提交任务：使用clas()函数通过submit提交到独立进程
                提交的任务必须是简单函数，进程并行不支持类方法、闭包等，
                函数参数和返回值必须兼容pickle序列化，因为进程间的通信需要传递可序列化对象
            """
            future_result = pool.submit(calc,keep_stock_threshold,buy_change_threshold)
            #当进程完成任务即calc运行结束后的回调函数
            future_result.add_done_callback(when_done)
    print(sorted(result)[::-1][:10])

# if __name__ == '__main__':
#     main()

#使用多线程
from concurrent.futures import ThreadPoolExecutor
result =[]
with ThreadPoolExecutor() as pool:
    for keep_stock_threshold, buy_change_threshold in itertools.product(keep_stock_list,buy_change_list):
        future_result = pool.submit(calc,keep_stock_threshold,buy_change_threshold)
        future_result.add_done_callback(when_done)
print(sorted(result)[::-1][:10])

import logging
#设置日志级别为info
logging.basicConfig(level=logging.INFO)

