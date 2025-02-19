'''
数据处理，使得输出格式为：
项目名称1(str)
    {
/一人+组参赛总数n(int)
/牌(str) 1 1
/同上 2 1
/。。。。。。
/同上 n-1 0
}
    {
/一人+组
/牌 1 1
/牌 2 0
}
项目名称2
。。。。。。
'''
import pandas as pd

# 读取CSV文件
df = pd.read_csv('summerOly_athletes_for_1.csv')

# 将Year列强制转换为int类型
df['Year'] = df['Year'].astype(int)

# 按sport分组
grouped_by_sport = df.groupby('Sport')

# 打开输出文件
with open('1_1_01_output_sep.txt', 'w', encoding='utf-8') as f:
    # 遍历每个sport
    for sport, sport_group in grouped_by_sport:
        # 写入sport名称
        f.write(f"{sport}\n")
        
        # 按event和name分组
        grouped_by_event_name = sport_group.groupby(['Event', 'Name'])
        
        # 遍历每个event-name配对
        for (event, name), group in grouped_by_event_name:
            # 按Year排序
            group_sorted = group.sort_values(by='Year')
            
            # 获取总记录行数
            n = len(group_sorted)
            
            # 写入数据块内容
            for i, (index, row) in enumerate(group_sorted.iterrows(), start=1):
                # 第一列是medal，第二列是序号，第三列是1或0
                medal = row['Medal']
                second_col = i
                third_col = 1 if i < n else 0
                f.write(f"{medal} {second_col} {third_col}\n")