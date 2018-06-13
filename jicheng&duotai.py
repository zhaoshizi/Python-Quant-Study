import six
from abc import ABCMeta, abstractmethod
import test2
from  functools import reduce

class TradeStrategyBase(six.with_metaclass(ABCMeta, object)):
    """
    交易策略抽象基类
    """
    """
    元类就是创建类的类，在Python中，只有type类及其子类才可以当元类，在一个类中可以用__metaclass__用于指定元类
    type创建类时，参数格式如下，classname是类名，字符串类型，parentclasses是类所有父类，元组类型，attrs是类的所有{属性:值}，
    type(classname, parentclasses , attrs)
    一切类的创建最终都会调用type.__new__(cls, classname, bases, attrs)，它会在堆中创建一个类对象，并返回
    当通过type.__new__(cls, classname, bases, attrs)创建类时，cls就是该类的元类，它是type类或其子类。
    ABCMeta是一个抽象类的元类，用来创建抽象类
    six.with_metaclass是用元类来创建类的方法，调用一个内部类，在内部类的__new__函数中，返回一个新创建的临时类
    six.with_metaclass(ABCMeta, object)就通过内部类的__new__函数返回一个ABCMeta元类创建的临时类
    作为TradeStrategyBase类的基类
    print(type(TradeStrategyBase))
    print(TradeStrategyBase.__class__)
    在python里类也是对象，上面两个语句可以看TradeStrategyBase类的类型，都是<class 'abc.ABCMeta'>，
    即TradeStrategyBase类是ABCMeta元类的对象，是由ABCMeta元类生成的。
    """

    @abstractmethod
    def buy_strategy(self, *args, **kwargs):
        #买入策略
        #pass为空语句，占位用
        pass

    @abstractmethod
    def sell_strategy(self, *args, **kwargs):
        #卖出策略基类
        pass


class TradeStrategy1(TradeStrategyBase):
    """
    交易策略1：追涨策略，当股份上涨一个阀值默认为7%时
    买入股票并持有s_keep_stock_threshold（20）天
    """
    s_keep_stock_threshold = 20

    def __init__(self):
        self.keep_stock_day = 0
        #7%上涨幅度作为买入策略阀值
        self.__buy_change_threshold = 0.07

    def buy_strategy(self, trade_ind, trade_day, trade_days):
        if self.keep_stock_day == 0 and trade_day.change > self.__buy_change_threshold:
            #当没有持有股票的时候self.keep_stock_day ==0 并且符合买入条件上涨一个阀值，买入
            self.keep_stock_day += 1
        elif self.keep_stock_day > 0:
            #self.keep_stock_day > 0代表持有股票，持有股票天数递增
            self.keep_stock_day += 1

    def sell_strategy(self, tarde_ind, trade_day, trade_days):
        if self.keep_stock_day >= TradeStrategy1.s_keep_stock_threshold:
            #当持有股票天数超过阀值s_keep_stock_threshold，卖出股票
            self.keep_stock_day = 0

    @property
    def buy_change_threshold(self):
        return self.buy_change_threshold

    @buy_change_threshold.setter
    def buy_change_threshold(self, buy_change_threshold):
        if not isinstance(buy_change_threshold, float):
            """
            上涨阀值需要为float类型
            """
            raise TypeError('buy_change_threshold must be float!')
        #上涨阀值只取小数点后两位
        self.__buy_change_threshold = round(buy_change_threshold, 2)

class TradeLoopBack(object):
    """
    交易回测系统
    """
    def __init__(self,trade_days,trade_strategy):
        """
        使用前面封装的StockTradeDays类和交易策略类TradeStrategyBase类初始化交易系统
        :param trade_days: StockTradeDays交易数据序列
        :param trade_strategy:TradeStrategyBase交易策略
        """
        self.trade_days = trade_days
        self.trade_strategy = trade_strategy
        #交易盈亏结果序列
        self.profit_array=[]

    def execute_trade(self):
        """
        执行交易回测
        :return:
        """
        for ind,day in enumerate(self.trade_days):
            """
            以时间为驱动，完成交易回测
            """
            if self.trade_strategy.keep_stock_day >0:
                #如果有持有股票，加入交易盈亏结果序列
                self.profit_array.append(day.change)
            #hasattr：用来查询对象有没有实现某个方法
            if hasattr(self.trade_strategy,'buy_strategy'):
                #买入策略执行
                self.trade_strategy.buy_strategy(ind,day,self.trade_days)
            if hasattr(self.trade_strategy,'sell_strategy'):
                #卖出策略执行
                self.trade_strategy.sell_strategy(ind,day,self.trade_days)


trade_loop_back = TradeLoopBack(test2.trade_days,TradeStrategy1())
trade_loop_back.execute_trade()
print('回测策略1总盈亏为:{}%'.format(reduce(lambda a,b:a+b,trade_loop_back.profit_array)*100))