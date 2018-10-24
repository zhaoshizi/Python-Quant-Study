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

# 将数据集转换为一个矩阵，以电影标题为列，以userId为索引，以评分为值
movie_matrix = df.pivot_table(index='userId',columns='title',values='rating')
print(movie_matrix.head())

# 使用sort_values工具，按评分次数降序排列
ratings.sort_values('number_of_ratings', ascending = False)
print(ratings.head())

AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']
print(AFO_user_rating.head())
print(contact_user_rating.head())

# 使用corrwith功能计算两个dataframe对象的行或列的两两相关关系
similar_to_air_force_one = movie_matrix.corrwith(AFO_user_rating)
print(similar_to_air_force_one.head())

# 计算《超时空接触》和其他电影之间的相关性
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
print(similar_to_contact.head())

corr_contact = pd.DataFrame(similar_to_contact,columns=['Correlation'])
corr_contact.dropna(inplace = True)
print(corr_contact.head())

corr_AFO= pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace = True)
print(corr_AFO.head())

# 加入ratings['number_of_ratings']列
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])
print(corr_AFO.head())
print(corr_contact.head())

corr_AFO_fix = corr_AFO[corr_AFO['number_of_ratings']>100].sort_values(by='correlation',ascending=False)
print(corr_AFO_fix.head())

corr_contact_fix = corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation',ascending=False)
print(corr_contact_fix.head())

corr_contact_and_AFO = corr_contact_fix.merge(corr_AFO_fix['Correlation'])