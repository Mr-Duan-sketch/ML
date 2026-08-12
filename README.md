# ML — 机器学习实战练习

基于 Python（scikit-learn、pandas、matplotlib 等）的机器学习实战项目集合，涵盖数据预处理、特征工程、模型训练、超参数调优与模型评估全流程。

---

## 目录

| 编号 | 项目 | 算法 | 简介 |
|---|---|---|---|
| 01 | [基于随机森林的天气预测](01_基于随机森林的天气预测.py) | RandomForestRegressor | 温度预测回归任务，含数据可视化、特征工程、RandomizedSearchCV + GridSearchCV 双重超参数调优 |

---

## 01 — 基于随机森林的天气预测

### 1. 项目概述

使用历史气温数据预测当日最高温度（`actual`），属于**回归**任务。模型为 `sklearn.ensemble.RandomForestRegressor`，通过随机搜索 + 网格搜索两阶段超参数调优寻找最优模型。

- **数据文件**：`data/temps.csv`（约 348 条记录）
- **特征**：`year`, `month`, `day`, `week`, `temp_2`（前天温度）, `temp_1`（昨天温度）, `average`（历史同期均值）, `friend`（朋友猜测值）
- **标签**：`actual`（当日实际最高温度）

### 2. 代码流程

#### 2.1 数据读取与预处理

```python
features = pd.read_csv('data/temps.csv')
```

- 将 `year`、`month`、`day` 拼接为 `datetime` 格式日期，用于后续画图
- 对 `week` 列进行**独热编码**（`pd.get_dummies`），将星期几转换为哑变量
- 划分训练集 / 测试集：`test_size=0.2, random_state=42`

#### 2.2 数据可视化

绘制 4 个子图，展示 `actual`、`temp_1`、`temp_2`、`friend` 随时间的变化趋势：

```python
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# actual / temp_1 / temp_2 / friend 四条曲线
```

#### 2.3 基础模型训练（已注释，可用于快速验证）

```python
# estimator = RandomForestRegressor(n_estimators=1000, random_state=42)
# estimator.fit(train_features, train_labels)
# predictions = estimator.predict(test_features)
# rmse = np.sqrt(np.mean((predictions - test_labels) ** 2))
```

默认参数（`n_estimators=1000`）下的 RMSE 可作为后续调优的**基线参考**。

#### 2.4 决策树可视化（已注释，按需启用）

```python
# tree = estimator.estimators_[5]            # 导出第 6 棵树
# export_graphviz(tree, out_file='tree.dot', feature_names=features_names, ...)
# graph.write_png('tree.png')
```

将随机森林中的单棵决策树导出为 DOT → PNG，直观理解树的分裂逻辑。

#### 2.5 特征重要性分析（已注释，按需启用）

```python
# importances = estimator.feature_importances_
# for feature, importance in sorted(zip(features_names, importances), ...):
#     print(f'{feature}: {importance}')
```

输出每个特征对预测的贡献权重，辅助特征筛选。

#### 2.6 简化模型（已注释，按需启用）

```python
# important_index = [features_names.index('temp_1'), features_names.index('average')]
# train_features = train_features[:, important_index]
# test_features = test_features[:, important_index]
```

仅使用最重要的两个特征（`temp_1` + `average`）重新训练，对比降维前后的精度变化。

#### 2.7 特征重要性条形图（已注释，按需启用）

```python
# plt.bar(range(len(importances)), importances)
# plt.xticks(range(len(importances)), features_names, rotation=90)
```

可视化特征权重排序。

#### 2.8 第一阶段：RandomizedSearchCV 随机搜索

在大范围内快速定位较优参数区域：

| 参数 | 搜索范围 |
|---|---|
| `n_estimators` | 200 ~ 2000（10 个等间距值） |
| `max_features` | `'auto'`, `'sqrt'` |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `bootstrap` | True, False |

- 迭代 100 次，3 折交叉验证
- 找到的最优参数：`{n_estimators: 1000, min_samples_split: 2, min_samples_leaf: 2, max_features: 'sqrt', max_depth: 20, bootstrap: False}`

#### 2.9 第二阶段：GridSearchCV 网格搜索

围绕 RandomSearchCV 最优参数，在更窄范围内精细遍历：

| 参数 | 候选值 | 数量 |
|---|---|---|
| `n_estimators` | 800, 1000, 1200, 1500 | 4 |
| `max_depth` | 15, 18, 20, 25, None | 5 |
| `min_samples_split` | 2, 3, 5 | 3 |
| `min_samples_leaf` | 1, 2, 4 | 3 |
| `max_features` | `'sqrt'`, `'log2'` | 2 |
| `bootstrap` | False（固定） | 1 |

- **360 种组合** × 5 折交叉验证 = 1800 次拟合
- `scoring='neg_mean_squared_error'`，`n_jobs=-1` 全核并行

### 3. 运行方式

```bash
cd ML
python "01_基于随机森林的天气预测.py"
```

> **提示**：如需启用被注释的代码块（基础模型、决策树可视化、特征重要性分析等），取消对应代码块的注释即可。各模块相互独立，可按需组合运行。

### 4. 项目结构

```
ML/
├── data/
│   └── temps.csv                            # 气温数据集
├── 01_基于随机森林的天气预测.py              # 主代码
├── tree.dot                                 # 决策树 DOT 文件（运行后生成）
├── tree.png                                 # 决策树可视化图片（运行后生成）
└── README.md                                # 本文件
```

---

## 后续项目

> 后续项目将陆续添加到此仓库，统一列于上方目录表中。

