import pandas as pd

# 读取原始数据
df = pd.read_csv("summerOly_athletes_1990y.csv")

# 筛选2020年和2024年的数据
df_2020 = df[df['Year'] == 2020][['NOC', 'Name', 'Division']]  # 2020年参赛运动员
df_2024 = df[df['Year'] == 2024][['NOC', 'Name', 'Division', 'Medal']]  # 2024年参赛运动员及奖牌信息

# 找出2024年的老运动员（即在2020年也参赛的运动员）
df_2020_athletes = df_2020.drop_duplicates(subset=['NOC', 'Name'])  # 去重
df_2024_athletes = df_2024.drop_duplicates(subset=['NOC', 'Name'])  # 去重

# 合并2020年和2024年的数据，找出老运动员
old_athletes = pd.merge(
    df_2024_athletes,
    df_2020_athletes,
    on=['NOC', 'Name'],
    how='inner',  # 只保留2020年和2024年都参赛的运动员
    suffixes=('', '_2020')
)

# 筛选出获奖的老运动员
# 筛选出获奖的老运动员
old_athletes_medals = old_athletes.dropna(subset=['Medal'])

# 计算 Division 的倒数并添加到 DataFrame 中
old_athletes_medals['Division_Reciprocal'] = 1 / old_athletes_medals['Division']

# 按国家和奖牌类型分组，计算 Division 倒数的加权和
medal_counts = (
    old_athletes_medals.groupby(['NOC', 'Medal'])['Division_Reciprocal']
    .sum()
    .unstack(fill_value=0)
)

# 重命名列
medal_counts = medal_counts.rename(columns={'Gold': 'Gold', 'Silver': 'Silver', 'Bronze': 'Bronze'})

# 填充缺失的奖牌列为0
if 'Gold' not in medal_counts.columns:
    medal_counts['Gold'] = 0
if 'Silver' not in medal_counts.columns:
    medal_counts['Silver'] = 0
if 'Bronze' not in medal_counts.columns:
    medal_counts['Bronze'] = 0

# 选择需要的列
medal_counts = medal_counts[['Gold', 'Silver', 'Bronze']].reset_index()

# 按金牌、银牌、铜牌的顺序对国家进行排序
medal_counts = medal_counts.sort_values(by=['Gold', 'Silver', 'Bronze'], ascending=[False, False, False])

# 保存到CSV文件
medal_counts.to_csv("old_athletes_medals_2024_sorted.csv", index=False)

# 打印结果
print(medal_counts)