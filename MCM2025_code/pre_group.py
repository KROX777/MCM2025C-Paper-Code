'''
加一列:一个牌子几个人分
'''
import pandas as pd

# 定义多人项目的Event列表
team_events = [
    "Athletics Men's 4 x 100 metres Relay",
    "Athletics Men's 4 x 400 metres Relay",
    "Athletics Women's 4 x 100 metres Relay",
    "Athletics Women's 4 x 400 metres Relay",
    "Basketball Men's Basketball",
    "Basketball Women's Basketball",
    "Beach Volleyball Men's Beach Volleyball",
    "Beach Volleyball Women's Beach Volleyball",
    "Cycling Men's Team Pursuit, 4,000 metres",
    "Cycling Women's Team Pursuit",
    "Equestrian Mixed Three-Day Event, Team",
    "Equestrian Mixed Jumping, Team",
    "Fencing Men's Foil, Team",
    "Fencing Men's epee, Team",
    "Fencing Women's Foil, Team",
    "Fencing Women's Sabre, Team",
    "Football Men's Football",
    "Football Women's Football",
    "Gymnastics Men's Team All-Around",
    "Gymnastics Women's Team All-Around",
    "Handball Men's Handball",
    "Handball Women's Handball",
    "Hockey Men's Hockey",
    "Hockey Women's Hockey",
    "Rowing Men's Coxed Eights",
    "Rowing Men's Coxed Fours",
    "Rowing Men's Coxless Fours",
    "Rowing Men's Coxless Pairs",
    "Rowing Men's Double Sculls",
    "Rowing Men's Lightweight Double Sculls",
    "Rowing Men's Quadruple Sculls",
    "Rowing Women's Coxed Eights",
    "Rowing Women's Coxed Fours",
    "Rowing Women's Coxless Fours",
    "Rowing Women's Coxless Pairs",
    "Rowing Women's Double Sculls",
    "Rowing Women's Lightweight Double Sculls",
    "Rowing Women's Quadruple Sculls",
    "Rugby Men's Rugby",
    "Rugby Women's Rugby",
    "Sailing Men's Two Person Dinghy - 470 Team",
    "Sailing Women's Two Person Dinghy - 470 Team",
    "Swimming Men's 4 x 100 metres Freestyle Relay",
    "Swimming Men's 4 x 200 metres Freestyle Relay",
    "Swimming Men's 4 x 100 metres Medley Relay",
    "Swimming Women's 4 x 100 metres Freestyle Relay",
    "Swimming Women's 4 x 200 metres Freestyle Relay",
    "Swimming Women's 4 x 100 metres Medley Relay",
    "Synchronized Swimming Women's Duet",
    "Synchronized Swimming Women's Team",
    "Table Tennis Men's Doubles",
    "Table Tennis Women's Doubles",
    "Table Tennis Men's Team",
    "Table Tennis Women's Team",
    "Volleyball Men's Volleyball",
    "Volleyball Women's Volleyball",
    "Water Polo Men's Water Polo",
    "Water Polo Women's Water Polo",
    "Baseball Men's Baseball",
    "Softball Women's Softball",
    "Gymnastics Women's Group",
    "Archery Women's Team",
    "Athletics Women's 1,500 metres",
    "Canoeing Men's Kayak Fours, 1,000 metres",
    "Canoeing Women's Kayak Fours, 500 metres",
    "Cycling Men's Team Sprint",
    "Cycling Women's Team Pursuit",
    "Cycling Women's Team Sprint",
    "Equestrian Mixed Jumping, Individual",
    "Fencing Men's Sabre, Team",
    "Gymnastics Men's Horse Vault",
    "Gymnastics Men's Rings",
    "Gymnastics Women's Balance Beam",
    "Gymnastics Women's Floor Exercise",
    "Gymnastics Women's Horse Vault",
    "Gymnastics Women's Individual All-Around",
    "Rowing Men's Lightweight Coxless Fours",
    "Sailing Women's Three Person Keelboat",
    "Tennis Women's Doubles",
    "Women's Canoe Double 500m Team",
    "Badminton Mixed Doubles",


]

# 读取CSV文件
df = pd.read_csv('summerOly_athletes_1990y.csv')

# 将Year列强制转换为int类型
df['Year'] = df['Year'].astype(int)

# 初始化Division列，默认值为1
df['Division'] = 1

# 筛选出Event属于team_events的行
team_event_df = df[df['Event'].isin(team_events)]

# 按除Name外的所有列分组
grouped = team_event_df.groupby(['Sex', 'Team', 'NOC', 'Year', 'City', 'Sport', 'Event', 'Medal'])

# 遍历每个分组
for group_key, group_data in grouped:
    # 获取当前分组的行数（即参加人数）
    division_size = len(group_data)
    
    # 将Division列的值设置为当前分组的行数
    df.loc[group_data.index, 'Division'] = division_size

# 导出到CSV文件
df.to_csv('summerOly_athletes_1990y.csv', index=False)

print("处理完成，结果已导出到 out.csv")