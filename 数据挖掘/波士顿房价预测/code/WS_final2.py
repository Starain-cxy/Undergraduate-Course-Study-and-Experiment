import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

filename = r".\boston_housing.csv"
df = pd.read_csv(filename)

print("缺失值检查")
if df.isnull().sum().sum() == 0:
    print("无缺失值")
    print(f"数据形状: {df.shape}")

print("\n离群值处理")
df_clean = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if col in ['chas', 'rad']:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    df_clean[col] = np.clip(df[col], lower, upper)

print("离群值已处理")

X = df_clean.drop('medv', axis=1)
y = df_clean['medv']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=34)
train_idx, test_idx = next(sss.split(X, X['chas']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\n数据划分:")
print(f"   训练集: {X_train.shape[0]} 条")
print(f"   测试集: {X_test.shape[0]} 条")
print(f"   训练集 chas=1: {X_train['chas'].mean():.2%}")
print(f"   测试集 chas=1: {X_test['chas'].mean():.2%}")

tree = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2']
}

print("\n执行超参数调优...")
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_tree = grid_search.best_estimator_
best_params = grid_search.best_params_
best_mse_cv = -grid_search.best_score_

print(f"最优参数: {best_params}")
print(f"5折CV平均MSE: {best_mse_cv:.4f}")

y_pred = best_tree.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

y_train_pred = best_tree.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"训练集 MSE = {train_mse:.4f}")
print(f"训练集 R² = {train_r2:.4f}")

print(f"\n测试集性能:")
print(f"   MSE = {test_mse:.4f}")
print(f"   R² = {test_r2:.4f}")

residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('CART决策树残差图')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

importances = best_tree.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
plt.title('特征重要性')
plt.ylabel('重要性得分')
plt.tight_layout()
plt.show()

print("\n特征重要性排序:")
for i in indices:
    print(f"{features[i]:15}: {importances[i]:.4f}")