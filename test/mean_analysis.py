import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts 

# 1. 加载数据
df_virtual = pd.read_csv('virtual-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
df_ai16z = pd.read_csv('ai16z-1min.csv', parse_dates=['timestamp'], index_col='timestamp')

# 确保数据对齐
data = pd.concat([df_virtual['close'], df_ai16z['close']], axis=1)
data.columns = ['virtual_close', 'ai16z_close']
data = data.dropna()
test_result = ts.coint(data['virtual_close'], data['ai16z_close'])
print(f"Cointegration test p-value: {test_result[1]}")

# 2. 协整性检验
def adf_test(series):
    result = adfuller(series)
    return result[1]  # 返回 p 值

# 检验价格序列是否协整
p_value = adf_test(data['ai16z_close'] - data['virtual_close'])
print(f"ADF Test p-value: {p_value}")

if p_value < 0.05:
    print("两组资产协整")
else:
    print("两组资产不协整")

# 3. 计算比率
def diff(df_virtual, df_ai16z):
    ratio_values = df_ai16z['close'] / df_virtual['close']
    return ratio_values.dropna()

ratio = diff(df_virtual, df_ai16z)

# 4. 可视化比率
plt.figure(figsize=(12, 6))
plt.plot(ratio, label='AI16Z / Virtual Ratio', color='green')
plt.title('AI16Z to Virtual Coin Ratio')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.legend()
plt.show()

# 5. 距离度量（可以选择其他方法）
euclidean_distance = np.sqrt(np.sum((data['ai16z_close'] - data['virtual_close']) ** 2))
print(f"欧几里得距离: {euclidean_distance}")