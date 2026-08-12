import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

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

# #训练
# estimator.fit(train_features, train_labels)
#
# #预测
# predictions = estimator.predict(test_features)
# mse = np.mean((predictions - test_labels)**2)
# rmse = np.sqrt(mse)
# print('RMSE:', rmse)

#决策树可视化
# tree = estimator.estimators_[5]
# export_graphviz(tree, out_file = 'tree.dot', feature_names = features_names, rounded = True, precision = 1)
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')

# #特征的重要性
# importances = estimator.feature_importances_
# features_imoportances = [(feature, round(importance, 2)) for feature, importance in zip(features_names, importances)]
# features_imoportances = sorted(features_imoportances, key = lambda x: x[1], reverse = True)
# for feature, importance in features_imoportances:
#     print(f'{feature}: {importance}')
#
#
# #简化模型，只使用最重要的两个特征来训练
# important_index= [features_names.index('temp_1'), features_names.index('average')]
# train_features = train_features[:, important_index]
# test_features = test_features[:, important_index]
#
# estimator.fit(train_features, train_labels)
# predictions = estimator.predict(test_features)
# mse = np.mean((predictions - test_labels)**2)
# rmse = np.sqrt(mse)
# print('RMSE:', rmse)
#
# #绘制特征重要性的条形图
# plt.bar(range(len(importances)), importances)
# plt.xticks(range(len(importances)), features_names, rotation = 90)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.tight_layout()
# plt.show()

#参数优化
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
estimator_rs = RandomizedSearchCV(estimator = estimator, param_distributions = random_grid, n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = 1)
estimator_rs.fit(train_features, train_labels)

print(estimator_rs.best_params_)

# ==================== GridSearchCV 精细调优 ====================
# 围绕 RandomSearchCV 最优参数构建精细网格
param_grid = {
    'n_estimators': [800, 1000, 1200, 1500],
    'max_depth': [15, 18, 20, 25, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [False],
}

# 使用 RandomSearchCV 最优参数初始化基础模型
base_estimator = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=20,
    bootstrap=False,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=base_estimator,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(train_features, train_labels)

print("=" * 50)
print("GridSearchCV 最优参数:", grid_search.best_params_)
print("GridSearchCV 最优得分 (neg_MSE):", grid_search.best_score_)

# 使用最优模型在测试集上评估
best_model = grid_search.best_estimator_
predictions = best_model.predict(test_features)
mse = np.mean((predictions - test_labels) ** 2)
rmse = np.sqrt(mse)
print(f"测试集 RMSE: {rmse:.4f}")
