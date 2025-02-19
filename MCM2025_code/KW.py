import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
# 读取CSV文件
df = pd.read_csv('summerOly_athletes.csv')

# 按国家、运动员姓名、Sport和Event分组，统计出现次数
count_by_country_name_sport_event = df.groupby(['NOC', 'Name', 'Sport', 'Event']).size().reset_index(name='Count')

# 统计每个国家的总出现次数
total_counts_by_country = count_by_country_name_sport_event.groupby('NOC')['Count'].sum().reset_index(name='Total Count')

# 筛选总人次前20的国家
top_20_countries = total_counts_by_country.nlargest(20, 'Total Count')['NOC']

# 统计每个国家每种次数的出现次数
count_distribution_by_country = count_by_country_name_sport_event.groupby(['NOC', 'Count']).size().reset_index(name='Frequency')

# 合并总出现次数和每种次数的出现次数
count_distribution_by_country = pd.merge(count_distribution_by_country, total_counts_by_country, on='NOC')

# 计算每种次数所占的比例
count_distribution_by_country['Proportion'] = count_distribution_by_country['Frequency'] / count_distribution_by_country['Total Count']

# 筛选前20国家的数据
top_20_country_distribution = count_distribution_by_country[count_distribution_by_country['NOC'].isin(top_20_countries)]

# 绘制所有前20国家的分布图在同一张图上
plt.figure(figsize=(12, 8))

# 定义颜色和标记
colors = plt.cm.tab20.colors  # 使用tab20颜色映射
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X','o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']  # 定义不同的标记

# 遍历每个国家，绘制曲线
for idx, (country, group) in enumerate(top_20_country_distribution.groupby('NOC')):
    plt.plot(
        group['Count'], 
        group['Proportion'], 
        marker=markers[idx], 
        color=colors[idx], 
        label=country, 
        linestyle='-', 
        linewidth=2, 
        markersize=8
    )

# 添加标签和标题
plt.xlabel('Count (Number of Occurrences)', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.title('Count Distribution for Top 20 Countries', fontsize=14, fontweight='bold')

# 添加图例
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('count_distribution_top_20_countries.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()


# 构建列联表
contingency_table = top_20_country_distribution.pivot_table(
    index='NOC', 
    columns='Count', 
    values='Frequency', 
    fill_value=0
)

# 进行卡方检验
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 输出结果
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")



# 提取每个国家的Proportion数据
grouped_data = [group['Proportion'].values for name, group in top_20_country_distribution.groupby('NOC')]

# 进行Kruskal-Wallis检验
h_statistic, p_value = kruskal(*grouped_data)

# 输出结果
print(f"H-statistic: {h_statistic}")
print(f"P-value: {p_value}")