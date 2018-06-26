import numpy as np
import pandas as pd

#200只股票
stock_cnt = 200
#504个交易日
view_days = 504
#生成服从正态分布，均值期望=0，标准差=1的序列
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
#DataFrame.head()默认显示前5行数据
print(pd.DataFrame(stock_day_change).head())
print(pd.DataFrame(stock_day_change).head(10))
#print(pd.DataFrame(stock_day_change).head()[:10])

#print(stock_day_change.shape[0])
#200
#索引行列序列
#股票0->股票stock_day_change[0]
stock_symbols = ['股票 '+str(x) for x in range(stock_day_change.shape[0])]
#通过构造直接设置index参数，head(2)就显示两行
print(pd.DataFrame(stock_day_change,index = stock_symbols).head(2))