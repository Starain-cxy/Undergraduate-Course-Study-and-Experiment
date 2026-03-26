import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#读取数据
filename = r".\IRIS.csv"
df = pd.read_csv(filename)

#数据预处理
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['species'])

X = df.drop('species', axis=1)

#分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # 保持类别比例
)
#归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只在训练集上拟合
X_test_scaled = scaler.transform(X_test)       # 用训练集参数转换测试集
#训练SVM模型
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)


y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

#混淆矩阵
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

class_names = label_encoder.classes_

# 10. 打印结果
print("=" * 60)
print("超参数设置:")
print(f"  核函数: rbf, C: 1.0, gamma: 'scale'")
print()

print("训练集混淆矩阵 (类别顺序: {})".format(", ".join(class_names)))
print(train_cm)
print()

print("测试集混淆矩阵 (类别顺序: {})".format(", ".join(class_names)))
print(test_cm)
print()

print("训练集分类报告:")
print(classification_report(y_train, y_train_pred, target_names=class_names))
print()

print("测试集分类报告:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

#计算准确率
train_accuracy = svm_model.score(X_train_scaled, y_train)
test_accuracy = svm_model.score(X_test_scaled, y_test)

print("=" * 60)
print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

