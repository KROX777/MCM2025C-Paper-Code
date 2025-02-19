import pandas as pd
import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

target_year = args.predict_year

# 读取数据
data = pd.read_csv("summerOly_athletes_1990y.csv")
data = data[data['Year'] < target_year]

# 东道主字典
olympics = {
    1896: "GRC", 1900: "FRA", 1904: "USA", 1908: "GBR", 1912: "SWE", 1920: "BEL",
    1924: "FRA", 1928: "NLD", 1932: "USA", 1936: "DEU", 1948: "GBR", 1952: "FIN",
    1956: "AUS", 1960: "ITA", 1964: "JPN", 1968: "MEX", 1972: "DEU", 1976: "CAN",
    1980: "URS", 1984: "USA", 1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS",
    2004: "GRC", 2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN", 2024: "FRA",
    2028: "USA", 2032: "AUS"
}

# 提取新增人数
def calculate_newcomers(data):
    # 确保数据按照年份排序
    data = data.sort_values(["Year", "NOC", "Sport", "Event", "Division"])

    # 初始化结果列表
    results = []

    have_year = {}

    ini_grouped = data.groupby(["Sport", "Event"])
    for (sport, event), group in ini_grouped:
        group = group.sort_values("Year")
        have_year[(sport, event)] = sorted(set(group["Year"]))

    # 按 (NOC, Sport, Event) 分组计算
    grouped = data.groupby(["NOC", "Sport", "Event", "Division"])

    # 按年份进行处理
    for (noc, sport, event, division), group in grouped:
        group = group.sort_values("Year")  # 确保每组按年份排序

        seen_athletes = set()

        # 遍历每年，计算新增人数
        for year in have_year[(sport, event)]:
            print(noc, sport, event, year)
            year_data = group[group["Year"] == year]
            current_athletes = set(year_data["Name"])  # 获取当前年份的运动员集合

            previous_length = len(seen_athletes)
            seen_athletes = seen_athletes.union(current_athletes)
            new_length = len(seen_athletes)
            
            newcomers = new_length - previous_length

            # 若是东道主，调整新增人数
            if noc == olympics.get(year):
                newcomers = -1

            if (year - 4) not in have_year[(sport, event)]:
                newcomers = 0
            
            # 保存结果
            results.append({
                "NOC": noc, "Sport": sport, "Event": event, "Year": year, "Division": division,
                "Newcomers": newcomers
            })

    return pd.DataFrame(results)

# 计算新增人数
result = calculate_newcomers(data)

# 透视表，将 (NOC, Sport, Event) 作纵列，Year 作横列
pivot_result = result.pivot_table(
    index=["NOC", "Sport", "Event", "Division"],
    columns="Year",
    values="Newcomers",
    aggfunc="sum"  # 聚合函数
).reset_index()

year_columns = [col for col in pivot_result.columns if str(col).isdigit()]

# 对每个年份列的值进行处理
for col_idx, col in enumerate(year_columns):
    for row_idx in range(len(pivot_result)):
        if pivot_result.at[row_idx, col] == -1:  # 如果当前单元格是 -1
            # 获取左边和右边的值
            left_value = pivot_result.at[row_idx, year_columns[col_idx - 1]] if col_idx > 0 else np.nan
            right_value = pivot_result.at[row_idx, year_columns[col_idx + 1]] if col_idx < len(year_columns) - 1 else np.nan
            
            # 更新当前值
            if not np.isnan(left_value) and not np.isnan(right_value):
                pivot_result.at[row_idx, col] = (left_value + right_value) / 2
            elif not np.isnan(left_value):
                pivot_result.at[row_idx, col] = left_value
            elif not np.isnan(right_value):
                pivot_result.at[row_idx, col] = right_value
            else:
                # 如果左右值都不存在，则删除该值
                pivot_result.at[row_idx, col] = np.nan


# 保存排序后的结果到 CSV 文件
pivot_result.to_csv("1_3_01_newcomers.csv", index=False)

print("排序后的结果已保存到 newcomers.csv")
