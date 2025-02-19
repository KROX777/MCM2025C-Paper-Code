import pandas as pd
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

PREDICT_YEAR = args.predict_year

# 示例数据操作
print(f"Processing data for years before {PREDICT_YEAR}")

# 读取 CSV 文件
input_file = 'summerOly_athletes_1990y.csv'  # 替换为你的输入文件路径
output_file = '1_2_01_athlete_event_medal_year_sorted.csv'  # 替换为你的输出文件路径

df = pd.read_csv(input_file)

# 保留所需列，并按指定顺序排序
required_columns = ['Name', 'Sport', 'Event', 'NOC', 'Year', 'Medal', 'Division']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV 文件缺少以下所需列: {set(required_columns) - set(df.columns)}")

df = df[required_columns]  # 按列顺序筛选
df = df.sort_values(by=['Name', 'Sport', 'Event', 'Year'])  # 排序

df = df[df['Year'] < PREDICT_YEAR]

# 保存到新的 CSV 文件
df.to_csv(output_file, index=False)

print(f"处理后的数据已保存到 {output_file}")