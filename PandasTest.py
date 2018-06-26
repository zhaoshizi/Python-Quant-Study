import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#200只股票
stock_cnt = 200
#504个交易日
view_days = 504
#生成服从正态分布，均值期望=0，标准差=1的序列
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
#DataFrame.head()默认显示前5行数据
#print(pd.DataFrame(stock_day_change).head())
#print(pd.DataFrame(stock_day_change).head(10))
#print(pd.DataFrame(stock_day_change).head()[:10])

#print(stock_day_change.shape[0])
#200
#索引行列序列
#股票0->股票stock_day_change[0]
stock_symbols = ['股票 ' + str(x) for x in range(stock_day_change.shape[0])]
#通过构造直接设置index参数，head(2)就显示两行
print(pd.DataFrame(stock_day_change, index=stock_symbols).head(2))

#从2017-1-1向上时间递进，单位freq='1d'即1天
days = pd.date_range('2017-1-1', periods=stock_day_change.shape[1], freq='1d')
#股票0->股票stock_days_change.shape[0]
stock_symbols = ['股票 ' + str(x) for x in range(stock_day_change.shape[0])]
# 分别设置index和columns
df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
print(df.head(2))
#df做个转置
df = df.T
print(df.head())
#重新采样，以21天为周期，对21天内的时间求平均来重新塑造数据
df_20 = df.resample('21D', how='mean')
print(df_20.head())

df_stock0 = df['股票 0']
#打印df_stock0类型
print(type(df_stock0))
# 打印出Series的前5行数据，与DataFrame一致
print(df_stock0.head())
#pandas 在基于封装Numpy的数据操作上，还封装了Moatplotlib
df_stock0.cumsum().plot()
plt.show()

#resample()函数重采样，how=ohlc代表周期的open、high、low和close值
# 以5天为周期重采样（周k）
df_stock0_5 = df_stock0.cumsum().resample('5D', how='ohlc')
# 以21天为周期重采样（月k）
df_stock0_20 = df_stock0.cumsum().resample('20D', how='ohlc')
# 打印5天重采样
print(df_stock0_5.head())

#使用abu量化系统中的ABuMarketDrawing.plot_candle_stick()方法，画出k线图
# volume成交量通过np.random.random(len(df_stock0_5))生成随机数据填充
from abupy import ABuMarketDrawing
ABuMarketDrawing.plot_candle_stick(
    df_stock0_5.index,
    df_stock0_5['open'].values,
    df_stock0_5['high'].values,
    df_stock0_5['low'].values,
    df_stock0_5['close'].values,
    np.random.random(len(df_stock0_5)),
    None,
    'stock',
    day_sum=False,
    html_bk=False,
    save=False)
plt.show()