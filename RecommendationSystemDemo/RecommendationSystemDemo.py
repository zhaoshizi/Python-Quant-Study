import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
# 读入排名数据
# df = pd.read_csv('RecommendationSystemDemo/ml-latest-small/ratings.csv',sep='\t',names=['userId','movieId','rating','timestamp'])
df = pd.read_csv('RecommendationSystemDemo/ml-latest-small/ratings.csv')
print(df.head())
# 读入电影名字
movie_titles = pd.read_csv('RecommendationSystemDemo/ml-latest-small/movies.csv')
print(movie_titles.head())

# 在movieId列合并
df = pd.merge(df,movie_titles,on='movieId')
print(df.head())

print(df.describe())
# 求各影片评分的均值
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())
# 求各影片评分的次数
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

ratings['rating'].hist(bins=50)
plt.show()
ratings['number_of_ratings'].hist(bins=60)
plt.show()
# 使用seaborn绘制散点图，通过jointplot()函数实现.探索电影评分和被评分次数之间的关系
sns.jointplot(x='rating',y='number_of_ratings',data=ratings)
plt.show()

# 将数据集转换为一个矩阵，以电影标题为列，以user_id为索引，以评分为值
movie_matrix = df.pivot_table(index='user_id',columns='title',values='rating')
print(movie_matrix.head())

# 使用sort_values工具，按评分次数降序排列
ratings.sort_values('number_of_ratings', ascending = False)
print(ratings.head())

AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']
print(AFO_user_rating.head())
print(contact_user_rating.head())

# 使用corrwith功能计算两个


