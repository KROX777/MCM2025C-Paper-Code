import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# === 1. 数据加载与预处理 ===
# 假设数据格式为 ['Year', 'Country', 'Event', 'Score']
data = pd.read_csv('/Users/oscar/Desktop/OX/CS/OI/Workplace/MCM/summerOly_medal_counts copy.csv')  # 替换为实际路径

# 筛选某国家和项目的数据
country = 'Italy'
df = data[(data['NOC'] == country)].sort_values('Year')

# 提取时间序列
time_series = df['Total'].values
years = (df['Year'].values / 4).astype(int)

valid_indices = time_series <= 100

# 筛选出符合条件的年份和对应的数据
time_series = time_series[valid_indices]
years = years[valid_indices]


# === 2. 检查数据平稳性 ===
result = adfuller(time_series)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("时间序列非平稳，进行一阶差分...")
    diff_series = np.diff(time_series)  # 一阶差分
else:
    diff_series = time_series

# === 3. 绘制 ACF 和 PACF 图，选择参数 p 和 q ===
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_acf(diff_series, lags=10, ax=plt.gca())
plt.title("ACF")

plt.subplot(122)
plot_pacf(diff_series, lags=10, ax=plt.gca())
plt.title("PACF")
plt.show()

# 从 ACF 和 PACF 图中选择参数（例如 p=1, d=1, q=1）
p, d, q = 2, 1, 1

# === 4. 构建 ARIMA 模型并拟合 ===
model = ARIMA(time_series, order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# === 5. 模型诊断 ===
residuals = fitted_model.resid
plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.plot(residuals)
plt.title("Residuals")

plt.subplot(212)
plot_acf(residuals, lags=20, ax=plt.gca())
plt.title("Residuals ACF")
plt.show()

data_ori = pd.read_csv('/Users/oscar/Desktop/OX/CS/OI/Workplace/MCM/summerOly_medal_counts.csv')
df_ori = data_ori[(data_ori['NOC'] == country)].sort_values('Year')
time_series_ori = df_ori['Total'].values
years_ori = df_ori['Year'].values

# === 6. 预测未来表现 ===
forecast_steps = 3  # 预测未来5年
forecast = fitted_model.forecast(steps=forecast_steps)

# 打印预测结果
future_years = range(years[-1] + 1, years[-1] + 1 + forecast_steps)
future_years = [year * 4 for year in future_years]

for year, pred in zip(future_years, forecast):
    print(f"Year {year}: Predicted Score = {pred}")

# === 7. 可视化结果 ===
plt.figure(figsize=(10, 6))
plt.plot(years_ori, time_series_ori, 'o', label="Historical Data")

plt.plot(future_years, forecast, 'o', label="Forecast", color="orange")
plt.xlabel("Year")
plt.ylabel("Score")
plt.title(f"ARIMA Forecast for {country}")
plt.legend()
plt.show()