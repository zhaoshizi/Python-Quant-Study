import pandas as pd
import numpy as np
import warnings
import matplotlib as plt
import seaborn as sns

warnings.filterwarnings('ignore')
# 读入排名数据
df = pd.read_csv(u'ml-latest-small\ratings.csv',sep='\t',names=['userId','movieId','rating','timestamp'])
# 读入电影名字
movie_titles = pd.read_csv(u'ml-latest-small\movies.csv')
movie_titles.head()

# 在movieId列合并
df = pd.merge(df,movie_titles,on='movieId')
df.head()

df.describe()
# 求各影片评分的均值
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
# 求各影片评分的次数
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings.head()
# %matplotlib inline
ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=60)
# 使用seaborn绘制散点图，通过jointplot()函数实现.探索电影评分和被评分次数之间的关系
sns.jointplot(x='rating',y='number_of_ratings',data=ratings)


