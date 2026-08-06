import pandas as pd
import numpy as np

# 1. 读取 Excel 文件
df = pd.read_excel('你的上证指数数据.xlsx')

# 2. 确保数据按日期升序排列（若原始已排序可跳过）
df = df.sort_values('日期Date').reset_index(drop=True)

# 3. 计算日对数收益率：r_t = ln(P_t / P_{t-1})
df['log_return'] = np.log(df['收盘Close'] / df['收盘Close'].shift(1))

# 4. 丢弃缺失值（第一个交易日无法计算收益率）
valid_returns = df['log_return'].dropna()*100

# 5. 导出为单列 CSV（无标题，无索引）
valid_returns.to_csv('log_returns.csv', index=False, header=False)

print("成功导出 log_returns.csv，共 {} 条收益率数据。".format(len(valid_returns)))
