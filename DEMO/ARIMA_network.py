import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
# 连接数据库
#连接配置信息
config = {
          'host':'127.0.0.1',
          'port':3306,#MySQL默认端口
          'user':'root',#mysql默认用户名
          'password':'123456',
          'db':'bigdata',#数据库
          'charset':'utf8',
          'cursorclass':pymysql.cursors.DictCursor,
          }

con=pymysql.connect(**config)
# 执行SQL语句
cursor=con.cursor()
cursor.execute('set names utf8')
cursor.execute('set autocommit=1')

# 从源数据库读取数据，并且将预测数据全部清除
sql = "select produceTime ,network_http_response_time from system_network where id<120"
cursor.execute(sql)
result=cursor.fetchall()
deletesql="delete from predict_network_source"
cursor.execute(deletesql)
deletesql1="delete from predict_network"
cursor.execute(deletesql1)
cursor.close()
con.close()

# df=pd.DataFrame(result,index=None,columns=['tomcat_request_count'])
df=pd.DataFrame(result)
df['network_http_response_time']=df['network_http_response_time'].astype('float32')
# df['tomcat_request_count'].map(lambda x:('%.2f')%x)
df['type']=0

# 将过去的数据导入新的数据库中
##将数据写入mysql的数据库，但需要先通过sqlalchemy.create_engine建立连接,且字符编码设置为utf8，否则有些latin字符不能处理
# 将原有数据库的数据删除

yconnect = create_engine('mysql+pymysql://root:123456@localhost:3306/bigdata?charset=utf8')
pd.io.sql.to_sql(df, 'predict_network_source', yconnect, schema='bigdata', if_exists='append')

# 将原来的dataframe格式的数据转变为Series类型，并且为其添加时间索引。
ts = pd.Series(df['network_http_response_time'].values)
dates = pd.date_range('2018-08-16 15:50:00', periods=120, freq='T')
ts.index = dates
# df=df.astype('float32')
# print(ts)
# df.plot(figsize=(12, 8))
plt.plot(ts)
plt.show()

# 时间序列差分
#  ARIMA模型要求的是平稳型，如果是非平稳型的时间序列则需要先进行差分，得到平稳的时间序列，使用ARIMA(p,d,q)模型，其中d表示差分次数。
# 使用df.diff(1)实现一阶差分的效果

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(111)
ts1=ts.diff(1)
ts1.plot(ax=ax1)
plt.plot(ts1)
plt.show()

fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
ts2 = ts.diff(2)
ts2.plot(ax=ax2)
plt.plot(ts2)
plt.show()

# 合适的q 和 p值
# 选择合适的ARIMA模型-即选择ARIMA模型中的P和q值

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
f=plt.figure(facecolor='white')
ax1=f.add_subplot(211)
plot_acf(ts,lags=40,ax=ax1)
ax2=f.add_subplot(212)
plot_pacf(ts,lags=40,ax=ax2)
plt.show()

# 预测结果
import statsmodels.api as sm

#
arma_mod20 = sm.tsa.ARMA(ts,(8,5)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)

pre = arma_mod20.predict('2018-08-16 17:50:00', '2018-08-16 18:10:00', dynamic=True)
print(pre)

fig, ax = plt.subplots(figsize=(12, 8))
ax = ts.ix['2018-08-16 15:50:00':].plot(ax=ax)
fig = arma_mod20.plot_predict('2018-08-16 17:50:00', '2018-08-16 18:10:00', dynamic=True, ax=ax, plot_insample=False)
plt.show()

# 将得到的预测数据写入数据库中，首先将series格式的数据变为dataframe格式
dict_pre = {'produceTime':pre.index,'network_http_response_time':pre.values,'type':1}
df_pre = pd.DataFrame(dict_pre)
print(df_pre)

# 将得到的dataframe格式的数据再次写入数据库

yconnect = create_engine('mysql+pymysql://root:123456@localhost:3306/bigdata?charset=utf8')
pd.io.sql.to_sql(df_pre, 'predict_network', yconnect, schema='bigdata', if_exists='append' )
