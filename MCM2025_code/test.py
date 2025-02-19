import pandas as pd

# 读取数据
df = pd.read_csv('summerOly_athletes.csv')

# 筛选 2020 年和 2016 年的数据
filtered_df = df[df['Year'].isin([2016, 2012])]

# 找到 (Name, Sport, Event) 相同的记录
duplicates = filtered_df[filtered_df.duplicated(subset=['Name', 'Sport', 'Event'], keep=False)]

# 输出到文件
duplicates.to_csv('duplicates.csv', index=False)

print("筛选结果已保存到 duplicates.csv")