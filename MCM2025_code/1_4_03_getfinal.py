import pandas as pd
import numpy as np
import argparse
# 设置命令行参数
# parser = argparse.ArgumentParser(description="Filter data by year")
# parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
# args = parser.parse_args()

PREDICT_YEAR = 2028

# parser = argparse.ArgumentParser(description="Process hyperparameters for 1_4_01_getv.py")
# parser.add_argument('--a', type=float, required=True, help="Hyperparameter a")
# parser.add_argument('--b', type=float, required=True, help="Hyperparameter b")
# parser.add_argument('--c', type=float, required=True, help="Hyperparameter c")

lambda_gold = 0.05  # 你可以根据需要调整这个系数
lambda_silver = 0.37
lambda_bronze = 0.01

# 读取 summerOly_athletes_1990y.csv 文件
athletes_df = pd.read_csv('summerOly_athletes_1990y.csv')

# 过滤出 2012 年的 (Sport, Event) 组合
if PREDICT_YEAR <= 2024:
    events = athletes_df[athletes_df['Year'] == PREDICT_YEAR][['Sport', 'Event']].drop_duplicates()
else:
    events = athletes_df[athletes_df['Year'] == 2024][['Sport', 'Event']].drop_duplicates()

# 读取 1_4_02_medal_totals.csv 文件
medal_totals_df = pd.read_csv('1_4_02_medal_totals.csv')

# 读取 1_4_02_medal_averages.csv 文件
medal_averages_df = pd.read_csv('1_4_02_medal_averages.csv')

# 过滤出与 2012 年的 (Sport, Event) 组合匹配的行
filtered_medal_totals = medal_totals_df.merge(events, on=['Sport', 'Event'], how='inner')

# 找出不匹配的 (Sport, Event) 组合
unmatched_events = events[~events.set_index(['Sport', 'Event']).index.isin(
    medal_totals_df.set_index(['Sport', 'Event']).index)]

# 对于不匹配的 (Sport, Event) 组合，从 medal_averages.csv 中获取数据
new_rows = []
for _, row in unmatched_events.iterrows():
    sport = row['Sport']
    event = row['Event']
    
    # 获取所有 NOC
    nocs = medal_totals_df['NOC'].unique()
    
    for noc in nocs:
        # 查找对应的 (NOC, Sport) 数据
        avg_data = medal_averages_df[(medal_averages_df['NOC'] == noc) & (medal_averages_df['Sport'] == sport)]
        
        if not avg_data.empty:
            gold = avg_data['AverageGold'].values[0]
            silver = avg_data['AverageSilver'].values[0]
            bronze = avg_data['AverageBronze'].values[0]
            no_medal = avg_data['AverageNoMedal'].values[0]
            print("YES")
        else:
            gold = 0
            silver = 0
            bronze = 0
            no_medal = 0
            print("NO")
        
        new_rows.append({
            'NOC': noc,
            'Sport': sport,
            'Event': event,
            'Gold': gold,
            'Silver': silver,
            'Bronze': bronze,
            'NoMedal': no_medal
        })

# 将新行添加到 filtered_medal_totals 中
new_rows_df = pd.DataFrame(new_rows)
final_df = pd.concat([filtered_medal_totals, new_rows_df], ignore_index=True)

# 假设已经读取了数据到 final_df
# 创建国家权重字典（基于 NOC 国家代号）
country_weights = {
    'USA': 2, 'CHN': 1, 'JPN': 1, 'AUS': 1, 'FRA': 1,
    'NED': 1, 'GBR': 1, 'KOR': 1, 'ITA': 1, 'GER': 1,
    'NZL': 0.5, 'CAN': 0.5, 'UZB': 0.5, 'HUN': 0.5, 'ESP': 0.5,
    # 添加其他国家代号及权重
}

# 汇总数据
summary_df = final_df.groupby('NOC', as_index=False).agg({
    'Gold': 'sum',
    'Silver': 'sum',
    'Bronze': 'sum',
    'NoMedal': 'sum'
})

# 映射权重
summary_df['Weight'] = summary_df['NOC'].map(country_weights).fillna(0)




# 筛选符合条件的国家
weighted_df = summary_df[summary_df['Weight'] >= 1].copy()

# 加权计算
weighted_df['Bronze'] += weighted_df['NoMedal'] * lambda_bronze
weighted_df['Silver'] += weighted_df['Bronze'] * lambda_silver
weighted_df['Bronze'] *= (1 - lambda_silver)
weighted_df['Gold'] += weighted_df['Silver'] * lambda_gold
weighted_df['Silver'] *= (1 - lambda_gold)
weighted_df['Gold'] += weighted_df['Bronze'] * lambda_gold * lambda_silver
weighted_df['Bronze'] *= (1 - lambda_gold * lambda_silver)

# 更新原表
summary_df.update(weighted_df)

weighted_df = summary_df[summary_df['Weight'] == 0.5].copy()

# 加权计算
weighted_df['Bronze'] += weighted_df['NoMedal'] * lambda_bronze * 0.5
weighted_df['Silver'] += weighted_df['Bronze'] * lambda_silver
weighted_df['Bronze'] *= (1 - lambda_silver)
weighted_df['Gold'] += weighted_df['Silver'] * lambda_gold
weighted_df['Silver'] *= (1 - lambda_gold)
weighted_df['Gold'] += weighted_df['Bronze'] * lambda_gold * lambda_silver * 0.5
weighted_df['Bronze'] *= (1 - lambda_gold * lambda_silver * 0.5)

# 更新原表
summary_df.update(weighted_df)

weighted_df = summary_df[summary_df['Weight'] == 0].copy()

# 加权计算
weighted_df['Bronze'] += weighted_df['NoMedal'] * lambda_bronze * 0.05
weighted_df['Silver'] += weighted_df['Bronze'] * lambda_silver
weighted_df['Bronze'] *= (1 - lambda_silver)
weighted_df['Gold'] += weighted_df['Silver'] * lambda_gold
weighted_df['Silver'] *= (1 - lambda_gold)
weighted_df['Gold'] += weighted_df['Bronze'] * lambda_gold * lambda_silver
weighted_df['Bronze'] *= (1 - lambda_gold * lambda_silver)


summary_df.update(weighted_df)

weighted_df = summary_df[summary_df['Weight'] == 2].copy()
weighted_df['Gold'] *= 2
summary_df.update(weighted_df)


# 计算 TotalScore
summary_df['TotalScore'] = (
    summary_df['Gold'] + summary_df['Silver'] + summary_df['Bronze']
)


aggregated_df = pd.read_csv("1_4_01_aggregated_by_noc.csv")

# 合并两个 DataFrame
merged_df = pd.merge(
    summary_df,
    aggregated_df[['NOC', 'Gold', 'Silver', 'Bronze', 'NoMedal']],
    on="NOC",
    how='left',  # 保留 summary_df 的所有行
    suffixes=('_summary', '_aggregated')
)

# 对 Gold, Silver, Bronze, NoMedal 相加
merged_df['Gold'] = merged_df['Gold_summary'] + merged_df['Gold_aggregated'].fillna(0)
merged_df['Silver'] = merged_df['Silver_summary'] + merged_df['Silver_aggregated'].fillna(0)
merged_df['Bronze'] = merged_df['Bronze_summary'] + merged_df['Bronze_aggregated'].fillna(0)
merged_df['NoMedal'] = merged_df['NoMedal_summary'] + merged_df['NoMedal_aggregated'].fillna(0)

# 计算新的 TotalScore
merged_df['TotalScore'] = merged_df['Gold'] + merged_df['Silver'] + merged_df['Bronze']

# 删除中间计算列
merged_df = merged_df.drop(
    columns=['Gold_summary', 'Gold_aggregated', 
             'Silver_summary', 'Silver_aggregated', 
             'Bronze_summary', 'Bronze_aggregated',
             'NoMedal_summary', 'NoMedal_aggregated']
)

# 按 TotalScore 降序排列
merged_df = merged_df.sort_values(
    by=["Gold", "Silver", "Bronze"], 
    ascending=[False, False, False]
)

# 保存结果到新的 CSV 文件
output_file = "1_4_03_noc_summary_with_totalscore"+str(PREDICT_YEAR)+".csv"
merged_df.to_csv(output_file, index=False)

print(f"合并后的结果已保存到 {output_file}")