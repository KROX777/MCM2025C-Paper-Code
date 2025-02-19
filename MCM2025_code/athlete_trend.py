import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 假设 res 是你的变化数据
# 统计每个 length 的变化模式
length_stats = {}

# 读取CSV文件
df = pd.read_csv('/Users/oscar/Desktop/OX/CS/OI/Workplace/MCM/summerOly_athletes.csv', sep=',', encoding='utf-8-sig')  # 假设CSV文件是用制表符分隔的

# 统计每个运动员参加每个Event的次数
medal_scores = {"Gold": 3, "Silver": 2, "Bronze": 1, "No medal": 0}

# 添加奖牌分值列
df["Medal Score"] = df["Medal"].map(medal_scores)

# 按选手和项目分组，并按年份排序
df = df.sort_values(by=["Name", "Event", "Year"])
df["Participant ID"] = df.groupby(["Name", "Event"]).ngroup()  # 为每个选手+项目分配唯一ID

# 初始化结果数组
res = []

# 遍历每个选手+项目的组合
for participant_id, group in df.groupby("Participant ID"):
    group = group.sort_values(by="Year")  # 按年份排序
    changes = []
    for j in range(1, len(group)):
        prev_score = group.iloc[j - 1]["Medal Score"]
        curr_score = group.iloc[j]["Medal Score"]
        changes.append(curr_score - prev_score)  # 计算变化值
    res.append(changes)

for changes in res:
    length = len(changes)
    if length >= 7:
        continue
    if length not in length_stats:
        length_stats[length] = {
            "always_positive_or_zero": 0,
            "always_negative": 0,
            "positive_then_negative": 0,
            "negative_then_positive": 0,
            "others": 0
        }
    
    if all(x >= 0 for x in changes):  # 一直正或零
        length_stats[length]["always_positive_or_zero"] += 1
    elif all(x <= 0 for x in changes):  # 一直负或零
        length_stats[length]["always_negative"] += 1
    else:
        # 检查先正后负
        first_positive_index = next((i for i, x in enumerate(changes) if x > 0), None)
        if first_positive_index is not None and all(x <= 0 for x in changes[first_positive_index + 1:]):
            length_stats[length]["positive_then_negative"] += 1
        # 检查是否先负后正
        first_negative_index = next((i for i, x in enumerate(changes) if x < 0), None)
        if first_negative_index is not None and all(x >= 0 for x in changes[first_negative_index + 1:]):
            length_stats[length]["negative_then_positive"] += 1
        # 如果既不是一直正/负，也不是先正后负或先负后正，则归类为 others
        else:
            length_stats[length]["others"] += 1

# 将统计结果转换为 DataFrame
length_df = pd.DataFrame(length_stats).T.fillna(0)

# 计算每个 length 的总数，用于计算比例
length_df['total'] = length_df.sum(axis=1)
length_df = length_df.div(length_df['total'], axis=0).drop(columns='total')

# 绘制堆叠柱状图
plt.figure(figsize=(12, 6))
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999', '#c2c2f0']  # 定义每个类别颜色
bottom = np.zeros(len(length_df))  # 初始化堆叠底部

for i, category in enumerate(length_df.columns):
    plt.bar(length_df.index, length_df[category], bottom=bottom, color=colors[i], label=category)
    bottom += length_df[category].values

# 添加标签和标题
plt.title("Proportion of Change Patterns by Length")
plt.xlabel("Length")
plt.ylabel("Proportion")
plt.xticks(length_df.index)  # 确保 x 轴刻度正确
plt.legend(title="Change Patterns", bbox_to_anchor=(1.05, 1), loc="upper left")

# 显示比例值
for length in length_df.index:
    cumulative = 0
    for i, category in enumerate(length_df.columns):
        value = length_df.loc[length, category]
        if value > 0:  # 只显示非零比例
            plt.text(length, cumulative + value / 2, f"{value:.1%}", ha="center", va="center", fontsize=8)
        cumulative += value

# 调整布局并保存图表
plt.tight_layout()
plt.savefig("stacked_bar_chart.png")
plt.show()