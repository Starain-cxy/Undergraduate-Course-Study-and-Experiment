import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

filename = r".\boston_housing.csv"
df = pd.read_csv(filename)

print("缺失值检查")
missing_count = df.isnull().sum()
if missing_count.sum() == 0:
    print("无缺失值")
    print(f"数据集形状: {df.shape}")

print("\n特征与目标变量(medv)的相关系数")
correlations = df.corr()['medv'].sort_values(ascending=False)
print(correlations.to_string())

plt.figure(figsize=(12, 6))
colors = ['red' if x >= 0 else 'blue' for x in correlations.values]
bars = plt.bar(correlations.index, correlations.values, color=colors, edgecolor='black', alpha=0.7)
for bar, value in zip(bars, correlations.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + (0.01 if height >= 0 else -0.02),
             f'{value:.3f}',
             ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
plt.title('特征与房价(medv)的线性相关系数', fontsize=14)
plt.xlabel('特征', fontsize=12)
plt.ylabel('Pearson相关系数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, shrink=0.8)

plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")

plt.title('所有变量间的 Pearson 相关系数热力图', fontsize=14)
plt.tight_layout()
plt.show()

print("离群值检测报告 (IQR方法)")
df_winsorized = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_summary = []
print("列名 | 离群值数量 | 占比(%) | 下界 | 上界 | ")
print("-" * 70)

for col in numeric_cols:
    if col in ['chas', 'rad']:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outliers_mask.sum()
    outlier_percent = outlier_count / len(df) * 100
    if outlier_count > 0:
        df_winsorized.loc[df[col] < lower, col] = lower
        df_winsorized.loc[df[col] > upper, col] = upper
        outliers_summary.append((col, outlier_count, outlier_percent, lower, upper, outlier_count))
        print(f"{col:15} {outlier_count:8} {outlier_percent:8.2f} {lower:8.2f} {upper:8.2f} ")
    else:
        print(f"{col:15} {outlier_count:8} {outlier_percent:8.2f} {lower:8.2f} {upper:8.2f} ")

if outliers_summary:
    total_outliers = sum([count for _, count, _, _, _, _ in outliers_summary])
    total_points = len(df) * len(numeric_cols)
    print(f"\n处理汇总")
    print(f"检测到离群值的列数: {len(outliers_summary)}/{len(numeric_cols)}")
    print(f"总计离群值数量: {total_outliers}")
    print(f"离群值占比: {total_outliers / total_points * 100:.2f}%")

X = df_winsorized.drop('medv', axis=1)
y = df_winsorized['medv']

X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=34
)
print(f"\n数据划分:")
print(f"   训练+验证集大小: {X_train_full.shape[0]} 条")
print(f"   独立测试集大小: {X_test_final.shape[0]} 条")

columns_to_scale = X_train_full.columns.drop(['chas', 'rad'])
scaler = StandardScaler()

X_train_full_scaled = X_train_full.copy()
X_train_full_scaled[columns_to_scale] = scaler.fit_transform(X_train_full[columns_to_scale])

X_test_final_scaled = X_test_final.copy()
X_test_final_scaled[columns_to_scale] = scaler.transform(X_test_final[columns_to_scale])

print(f"\n标准化完成: {len(columns_to_scale)} 个连续特征")
print(f"   标准化特征: {list(columns_to_scale)}")

alphas = np.logspace(-3, 3, 50)
ridge = Ridge(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=ridge,
    param_grid={'alpha': alphas},
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

print("\n正在执行 5 折交叉验证...")
grid_search.fit(X_train_full_scaled, y_train_full)

best_alpha_cv = grid_search.best_params_['alpha']
best_mse_cv = -grid_search.best_score_
print(f"最优 alpha = {best_alpha_cv:.4f}")
print(f"平均验证 MSE = {best_mse_cv:.4f}")

final_model = Ridge(alpha=best_alpha_cv, random_state=42)
final_model.fit(X_train_full_scaled, y_train_full)

y_train_pred = final_model.predict(X_train_full_scaled)
train_mse = mean_squared_error(y_train_full, y_train_pred)
train_r2 = r2_score(y_train_full, y_train_pred)

print(f"\n训练+验证集性能:")
print(f"   MSE = {train_mse:.4f}")
print(f"   R² = {train_r2:.4f}")

y_test_pred = final_model.predict(X_test_final_scaled)
test_mse = mean_squared_error(y_test_final, y_test_pred)
test_r2 = r2_score(y_test_final, y_test_pred)

print(f"\n测试集性能:")
print(f"   MSE = {test_mse:.4f}")
print(f"   R² = {test_r2:.4f}")

cv_mse_scores = []
for alpha in alphas:
    model = Ridge(alpha=alpha, random_state=42)
    scores = cross_val_score(
        model, X_train_full_scaled, y_train_full,
        cv=kf, scoring='neg_mean_squared_error'
    )
    cv_mse_scores.append(-scores)
cv_mse_scores = np.array(cv_mse_scores)

mean_mse = cv_mse_scores.mean(axis=1)
std_mse = cv_mse_scores.std(axis=1)
best_idx = np.argmin(mean_mse)

plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(alphas, cv_mse_scores[:, i], 'gray', alpha=0.4, linewidth=1)
plt.plot(alphas, mean_mse, 'r-', linewidth=2.5, label=f'平均验证MSE')
plt.fill_between(alphas, mean_mse - std_mse, mean_mse + std_mse, color='red', alpha=0.1)
plt.axvline(alphas[best_idx], color='k', linestyle='--', alpha=0.8)

plt.xscale('log')
plt.xlabel('正则化系数 α (log scale)')
plt.ylabel('验证集 MSE')
plt.title('5 折交叉验证: α参数搜索')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame({
    'feature': X_train_full_scaled.columns,
    'coefficient': final_model.coef_,
    'abs_coef': np.abs(final_model.coef_)
}).sort_values('abs_coef', ascending=False)

intercept_row = pd.DataFrame({
    'feature': ['intercept'],
    'coefficient': [final_model.intercept_],
    'abs_coef': [np.abs(final_model.intercept_)]
})

coef_df = pd.concat([coef_df, intercept_row], ignore_index=True)

print(f"\n模型系数:")
print(coef_df[['feature', 'coefficient']].round(4).to_string(index=False))

residuals = y_test_final - y_test_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, alpha=0.7, edgecolors='k', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('预测值 (Predicted medv)')
plt.ylabel('残差 (Residuals)')
plt.title('残差图')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n残差统计:")
print(f"   均值: {np.mean(residuals):.6f}")
print(f"   标准差: {np.std(residuals):.4f}")