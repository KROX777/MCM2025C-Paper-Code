import pandas as pd

# 读取数据
file_path = '1_2_03_athlete_event_fitted_results_updated.csv'
df = pd.read_csv(file_path, delimiter=',')  # 如果分隔符是制表符

df = df.groupby(['Sport', 'NOC', 'First Year'], as_index=False).agg({
    'p_max': 'mean',
    'a_max': 'mean',
    'alpha': 'mean',
    'beta': 'mean'
})

# 按 (NOC, Sport, Event) 排序，再按 Year 升序排序
df_sorted = df.sort_values(by=['NOC', 'Sport', 'First Year'])

# 筛选所需列
columns_to_keep = ['Sport', 'NOC', 'First Year', 'p_max', 'a_max', 'alpha', 'beta']
df_filtered = df_sorted[columns_to_keep]

# 按 (Sport, Event, NOC, First Year) 分组，并对后面的四列取平均值


# 保存修改后的数据集为新的 CSV 文件
df_filtered.to_csv('1_2_04_average_values_by_noc_sport_final.csv', index=False)



print(f"处理后的数据已保存")