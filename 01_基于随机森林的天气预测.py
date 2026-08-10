import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 读取数据
features = pd.read_csv('data/temps.csv')
print(features.head(5))
print(features.shape)

# 日期数据格式的处理
years = features['year']
months = features['month']
days = features['day']
# 将日期格式化为%Y-%m-%d的格式
dates = [str(int( year))+'-'+str(int(month))+'-'+str(int(day))for year, month, day in zip(years, months, days)]
dates =[datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates[:5])

#数据可视化
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
fig.autofmt_xdate(rotation=45)
ax[0,0].plot(dates, features['actual'])
ax[0,0].set_xlabel('Date')
ax[0,0].set_ylabel('Temperature')

ax[0,1].plot(dates, features['temp_1'])
ax[0,1].set_xlabel('Date')
ax[0,1].set_ylabel('Temperature')

ax[1,0].plot(dates, features['temp_2'])
ax[1,0].set_xlabel('Date')
ax[1,0].set_ylabel('Temperature')

ax[1,1].plot(dates, features['friend'])
ax[1,1].set_xlabel('Date')
ax[1,1].set_ylabel('Temperature')
plt.tight_layout (pad=2)
plt.show()

#独热编码
features = pd.get_dummies(features)
print(features.head(5))

#特征与标签
labels = np.array(features['actual'])
features = features.drop('actual', axis = 1)
features_names = list(features.columns)
features = np.array(features)

#数据集划分
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)

#建模
estimator  = RandomForestRegressor(n_estimators = 1000, random_state = 42)

#训练
estimator.fit(train_features, train_labels)

#预测
predictions = estimator.predict(test_features)
mse = np.mean((predictions - test_labels)**2)
rmse = np.sqrt(mse)
print('RMSE:', rmse)