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
