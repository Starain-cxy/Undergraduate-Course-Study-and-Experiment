import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#读取数据
filename = r".\IRIS.csv"
df = pd.read_csv(filename)

#数据预处理
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['species'])

X = df.drop('species', axis=1)
feature_names = X.columns.tolist()
##分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
##归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练softmax模型
softmax_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=77
)
softmax_model.fit(X_train_scaled, y_train)

#预测、评估
y_train_pred = softmax_model.predict(X_train_scaled)
y_test_pred = softmax_model.predict(X_test_scaled)

class_names = label_encoder.classes_

#结果
print("=" * 60)
print("Softmax回归模型评估")
print("=" * 60)
print()

print("训练集混淆矩阵 (类别顺序: {})".format(", ".join(class_names)))
print(confusion_matrix(y_train, y_train_pred))
print()

print("测试集混淆矩阵 (类别顺序: {})".format(", ".join(class_names)))
print(confusion_matrix(y_test, y_test_pred))
print()

print("训练集分类报告:")
print(classification_report(y_train, y_train_pred, target_names=class_names))
print()

print("测试集分类报告:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

#准确率
train_accuracy = softmax_model.score(X_train_scaled, y_train)
test_accuracy = softmax_model.score(X_test_scaled, y_test)

print("=" * 60)
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
print()

#显示模型系数（权重）
# print("模型系数（权重）:")
# print(f"特征: {feature_names}")
# for i, class_name in enumerate(class_names):
#     print(f"类别 '{class_name}' 的系数:")
#     print(f"  截距: {softmax_model.intercept_[i]:.4f}")
#     for j, feat_name in enumerate(feature_names):
#         print(f"  {feat_name}: {softmax_model.coef_[i, j]:.4f}")
#     print()

# ==================== 绘制二维投影决策边界 ====================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义特征组合
feature_pairs = [
    ('petal_length', 'petal_width', '花瓣长度 vs 花瓣宽度'),
    ('sepal_length', 'petal_length', '花萼长度 vs 花瓣长度')
]

# 定义颜色和标记
colors = ['red', 'blue', 'green']
species_list = df['species'].unique()

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 遍历两组特征组合
for idx, (x_feat, y_feat, title) in enumerate(feature_pairs):
    ax = axes[idx]

    # 获取当前特征对的索引
    x_idx = feature_names.index(x_feat)
    y_idx = feature_names.index(y_feat)

    # 绘制每个类别的散点
    for i, species in enumerate(species_list):
        species_data = df[df['species'] == species]
        ax.scatter(
            species_data[x_feat],
            species_data[y_feat],
            c=colors[i],
            s=60,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8,
            label=species
        )

    # 设置标题和标签
    ax.set_title(f'{title} (Softmax决策边界投影)\n({x_feat} vs {y_feat})', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(x_feat.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_feat.replace('_', ' ').title(), fontsize=12)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    ax.legend(loc='upper left', fontsize=10)

    # ========== 绘制Softmax回归决策边界投影 ==========
    # 创建网格
    x_min, x_max = df[x_feat].min() - 0.5, df[x_feat].max() + 0.5
    y_min, y_max = df[y_feat].min() - 0.5, df[y_feat].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 为每个网格点创建完整的特征向量
    # 其他特征设为整个数据集的均值
    grid_points = np.zeros((xx.ravel().shape[0], len(feature_names)))

    # 设置当前特征对的值
    grid_points[:, x_idx] = xx.ravel()
    grid_points[:, y_idx] = yy.ravel()

    # 设置其他特征为均值
    for i, feat in enumerate(feature_names):
        if feat not in [x_feat, y_feat]:
            grid_points[:, i] = X[feat].mean()

    # 标准化网格点
    grid_points_scaled = scaler.transform(grid_points)

    # 预测概率
    Z_proba = softmax_model.predict_proba(grid_points_scaled)

    # 获取每个点的预测类别
    Z = np.argmax(Z_proba, axis=1)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和区域
    from matplotlib.colors import ListedColormap

    custom_cmap = ListedColormap([(1, 0.8, 0.8), (0.8, 0.8, 1), (0.8, 1, 0.8)])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=custom_cmap, levels=np.arange(-0.5, 3, 1))

    # 绘制决策边界线
    ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5, levels=np.arange(-0.5, 3, 1))

    # 计算并显示决策边界的数学表达式（二维投影）
    # 对于Softmax回归，每个类别的决策函数是线性函数
    # 我们计算每个类别对在二维平面上的决策边界
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            # 计算两个类别之间的决策边界系数
            # 在Softmax中，两个类别i和j的决策边界是：
            # (w_i - w_j)·x + (b_i - b_j) = 0
            w_diff = softmax_model.coef_[i] - softmax_model.coef_[j]
            b_diff = softmax_model.intercept_[i] - softmax_model.intercept_[j]

            # 在二维投影中，我们固定其他特征为均值
            # 决策边界方程简化为：w_diff[x_idx]*x + w_diff[y_idx]*y + C = 0
            # 其中 C = Σ_{k≠x_idx,y_idx} w_diff[k]*mean_k + b_diff

            # 计算常数项
            constant = b_diff
            for k, feat in enumerate(feature_names):
                if k not in [x_idx, y_idx]:
                    constant += w_diff[k] * (scaler.mean_[k] / scaler.scale_[k])

            # 绘制决策边界线（直线）
            if abs(w_diff[y_idx]) > 1e-10:  # 避免除零
                # 从直线方程解出y: y = (-w_diff[x_idx]*x - constant) / w_diff[y_idx]
                x_line = np.linspace(x_min, x_max, 100)
                y_line = (-w_diff[x_idx] * x_line - constant) / w_diff[y_idx]

                # 只绘制在显示范围内的线段
                valid_indices = (y_line >= y_min) & (y_line <= y_max)
                if np.any(valid_indices):
                    ax.plot(x_line[valid_indices], y_line[valid_indices],
                            color='black', linewidth=1, linestyle='--', alpha=0.7)

# 调整整体标题
fig.suptitle('鸢尾花数据集特征散点图分析（Softmax回归决策边界投影）', fontsize=16, fontweight='bold', y=1)

# 调整布局并显示
plt.tight_layout()
plt.show()

# # ==================== 三维可视化（可选） ====================
# # 如果你想查看三维决策边界，可以取消注释以下代码
#
# from mpl_toolkits.mplot3d import Axes3D
#
# # 选择一个特征对进行3D可视化
# from mpl_toolkits.mplot3d import Axes3D
# x_feat, y_feat, title = feature_pairs[0]
# x_idx = feature_names.index(x_feat)
# y_idx = feature_names.index(y_feat)
#
# fig3d = plt.figure(figsize=(10, 8))
# ax3d = fig3d.add_subplot(111, projection='3d')
#
# # 绘制3D散点图
# for i, species in enumerate(species_list):
#     species_data = df[df['species'] == species]
#     ax3d.scatter(
#         species_data[x_feat],
#         species_data[y_feat],
#         species_data['sepal_width'],  # 第三个特征
#         c=colors[i],
#         s=60,
#         alpha=0.7,
#         edgecolors='black',
#         linewidth=0.8,
#         label=species
#     )
#
# ax3d.set_xlabel(x_feat.replace('_', ' ').title())
# ax3d.set_ylabel(y_feat.replace('_', ' ').title())
# ax3d.set_zlabel('sepal_width')
# ax3d.set_title(f'3D特征可视化\n{x_feat} vs {y_feat} vs sepal_width')
# ax3d.legend()
#
# plt.tight_layout()
# plt.show()
