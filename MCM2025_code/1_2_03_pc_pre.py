import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ini_age = 16

# 加载数据
df = pd.read_csv('1_2_02_athlete_event_fitted_results.csv')
df_backup = pd.read_csv('1_2_02_athlete_event_fitted_results.csv') 
df = df[(df['p_max'].notna()) & (df['Age'] != ini_age)]
df_backup = df_backup[df_backup['p_max'].notna()]

# 计算每个 (NOC, Sport) 组合的 p_max, a_max, alpha, beta 的纵列平均值
average_values = df.groupby(['NOC', 'Sport'])[['a_max', 'alpha', 'beta']].mean().reset_index()

# 将平均值合并到备份数据中
df_backup = df_backup.merge(
    average_values, on=['NOC', 'Sport'], how='left',
    suffixes=('', '_avg')
)

# 更新 Age=8 的 p_max, a_max, alpha, beta
mask = df_backup['Age'] == ini_age
df_backup.loc[mask, 'a_max'] = df_backup.loc[mask, 'a_max_avg']
df_backup.loc[mask, 'alpha'] = df_backup.loc[mask, 'alpha_avg']
df_backup.loc[mask, 'beta'] = df_backup.loc[mask, 'beta_avg']

# 删除多余的列
df_backup.drop(columns=['a_max_avg', 'alpha_avg', 'beta_avg'], inplace=True)


# # 检查缺失值的列
missing_columns = ['alpha', 'beta', 'a_max']

# 初始化缺失值填补方法
def fill_missing_values(row):
    ini_age = row['Age']  # 从表格中获取初始年龄
    strengths = row['Strength']  # 获取对应的实力值
    p_max = row['p_max']  # 获取当前行的 p_max 值
    
    if pd.isna(row['alpha']):
        alpha = np.random.uniform(0.05, 1)
    else:
        alpha = row['alpha']
    
    if pd.isna(row['beta']):
        beta = np.random.uniform(0.05, 0.5)
    else:
        beta = row['beta']
    
    if pd.isna(row['a_max']):
        def equation(x):
            return (-alpha * np.log(x) + beta * x -
                    (alpha * np.log(ini_age) - beta * ini_age + np.log(strengths / p_max))) ** 2
        
        result = minimize_scalar(equation, bounds=(4, 16), method='bounded')
        
        if result.success:
            print(f"好")
            a_max = result.x
        else:
            a_max = np.random.uniform(4, 20)
            print(f"未找到合适的解 for row {row.name}, assigning random a_max.")
    else:
        a_max = row['a_max']
    
    return pd.Series({'alpha': alpha, 'beta': beta, 'a_max': a_max})

# 填补缺失值
df_backup[['alpha', 'beta', 'a_max']] = df_backup.apply(fill_missing_values, axis=1)


# 保存更新后的备份数据
df_backup.to_csv('1_2_03_athlete_event_fitted_results_updated.csv', index=False)

print("处理完成！更新后的数据已保存为 'athlete_event_fitted_results_backup_updated.csv'")