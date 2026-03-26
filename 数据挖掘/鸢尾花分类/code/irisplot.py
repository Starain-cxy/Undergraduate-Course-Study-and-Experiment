import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
#读取数据
filename=r".\IRIS.csv"
df = pd.read_csv(filename)


# 绘图寻找特征（此段代码来自Deepseek）
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
    ax.set_title(f'{title}\n({x_feat} vs {y_feat})', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(x_feat.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_feat.replace('_', ' ').title(), fontsize=12)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    ax.legend(loc='upper left', fontsize=10)

# 调整整体标题
fig.suptitle('鸢尾花数据集特征散点图分析', fontsize=16, fontweight='bold', y=1)

# 调整布局并显示
plt.tight_layout()
plt.show()



