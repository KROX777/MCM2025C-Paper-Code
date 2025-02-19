import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('/Users/oscar/Desktop/OX/CS/OI/Workplace/MCM/summerOly_athletes.csv', sep=',', encoding='utf-8-sig')  # 假设CSV文件是用制表符分隔的

# 统计每个运动员参加每个Event的次数
event_participation_count = df.groupby(['Name', 'Event']).size().reset_index(name='Participation Count')

# 按参与次数排序
event_participation_count = event_participation_count.sort_values(by='Participation Count', ascending=False)

# 统计参与i次的人数
participation_distribution = event_participation_count['Participation Count'].value_counts().sort_index().reset_index()
participation_distribution.columns = ['Participation Count', 'Number of Athletes']

event_participation_count.to_csv('event_participation_count.csv', index=False)

# 打印统计结果
print(participation_distribution)

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(participation_distribution['Participation Count'], participation_distribution['Number of Athletes'], color='skyblue')
plt.xlabel('Participation Count (i)')
plt.ylabel('Number of Athletes')
plt.title('Distribution of Athletes by Participation Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.show()