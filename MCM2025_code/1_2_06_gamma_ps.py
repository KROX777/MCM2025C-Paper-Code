import numpy as np
import pandas as pd

# 读取 CSV 文件
file_path = '1_2_03_athlete_event_fitted_results_updated.csv'
df = pd.read_csv(file_path)

# 提取 p_max, a_max, alpha, beta 四列
columns_of_interest = ['p_max', 'a_max', 'alpha', 'beta']
if not all(col in df.columns for col in columns_of_interest):
    raise ValueError(f"文件缺少以下必要列: {set(columns_of_interest) - set(df.columns)}")

data = df[columns_of_interest].values

# 转换为 NumPy 数组
data_array = np.array(data)

# 对每列求平均
column_means = np.nanmean(data_array, axis=0)  # 忽略 NaN 值计算平均

# 输出结果
print("每列的平均值为：", column_means)

# 保存到文件
output_file = '1_2_06_gamma_p(s).txt'
with open(output_file, 'w') as file:
    file.write(' '.join(f"{mean:.2f}" for mean in column_means))

print(f"平均值已保存到文件 {output_file}")