'''
写程序，我要从1_3_02_added_number.csv中对每个(NOC, Sport, Event)取它的'Predicted'这一列的数记为p，再在1_2_05_noc_sport_all_years_updated_models.csv中找到它的(NOC, Sport)所对应的p_max	a_max	alpha	beta，然后我用row['p_max'] * (t / row['a_max'])**row['alpha'] * np.exp(-row['beta'] * (t - row['a_max']))算出来一个result，再根据def map_result(result):
    if 0 <= result < 3.2:
        return 0
    elif 3.2 <= result < 5:
        return 1
    elif 5 <= result < 10:
        return 2
    else:
        return 3
对result进行映射，是0的话，这个(NOC, Sport, Event)的'NoMedal'+p，1的话，这个(NOC, Sport, Event)的'Bronze'+p，2的话，这个(NOC, Sport, Event)的'Silver'+p，3的话，这个(NOC, Sport, Event)的'Gold'+p
我开一个新csv，输出每个(NOC,Sport,Event)对应的'Gold','Silver','Bronze'的值
再开一个新csv，输出每个(NOC,Sport)对应的'Gold','Silver','Bronze'的平均值（即对每个Event对应的求平均）
'''
import pandas as pd
import numpy as np
import argparse
import sys

# # 设置命令行参数
# parser = argparse.ArgumentParser(description="Filter data by year")
# parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
# args = parser.parse_args()

PREDICT_YEAR = 2028

deltap = float(sys.argv[1])

ini_age = 16

def map_result(result):
    if 0 <= result < 2:
        return 0
    elif 2.2 <= result < 3.5:
        return 1
    elif 3.5 <= result < 6:
        return 2
    else:
        return 3

# 读取数据
added_number_file = "1_3_02_added_number.csv"
updated_models_file = "1_2_05_noc_sport_all_years_updated_models.csv"

# 加载数据
added_data = pd.read_csv(added_number_file)
models_data = pd.read_csv(updated_models_file)

# 初始化结果字典
medal_totals = {}
medal_averages = {}

# 遍历每个 (NOC, Sport, Event)
for _, row in added_data.iterrows():
    noc, sport, event = row["NOC"], row["Sport"], row["Event"]
    p = row["Predicted"] / row["Division"]

    # 查找对应的 (NOC, Sport) 参数
    model_row = models_data[(models_data["NOC"] == noc) & (models_data["Sport"] == sport)]
    if model_row.empty:
        # print("fuck")
        # print(noc, sport)
        continue  # 如果未找到对应参数，则跳过

    model_row = model_row.iloc[0]
    p_max = model_row["p_max"]
    a_max = model_row["a_max"]
    alpha = model_row["alpha"]
    beta = model_row["beta"]

    # 计算结果
    result = p_max * (ini_age / a_max) ** alpha * np.exp(-beta * (ini_age - a_max))
    mapped_result = map_result(result)
    print(p_max, a_max, alpha, beta, result, mapped_result)
    # 更新 medal_totals 字典
    key = (noc, sport, event)
    if key not in medal_totals:
        medal_totals[key] = {"Gold": 0, "Silver": 0, "Bronze": 0, "NoMedal": 0}

    if mapped_result == 0:
        medal_totals[key]["NoMedal"] += p+deltap
    elif mapped_result == 1:
        medal_totals[key]["Bronze"] += p
    elif mapped_result == 2:
        medal_totals[key]["Silver"] += p
    elif mapped_result == 3:
        medal_totals[key]["Gold"] += p

# 转换 medal_totals 为 DataFrame
medal_totals_df = pd.DataFrame(
    [
        {"NOC": noc, "Sport": sport, "Event": event, **values}
        for (noc, sport, event), values in medal_totals.items()
    ]
)

# 计算每个 (NOC, Sport) 的平均值
for (noc, sport), group in medal_totals_df.groupby(["NOC", "Sport"]):
    avg_gold = group["Gold"].mean()
    avg_silver = group["Silver"].mean()
    avg_bronze = group["Bronze"].mean()
    avg_no_medal = group["NoMedal"].mean()

    medal_averages[(noc, sport)] = {
        "AverageGold": avg_gold,
        "AverageSilver": avg_silver,
        "AverageBronze": avg_bronze,
        "AverageNoMedal": avg_no_medal
    }

# 转换 medal_averages 为 DataFrame
medal_averages_df = pd.DataFrame(
    [
        {"NOC": noc, "Sport": sport, **values}
        for (noc, sport), values in medal_averages.items()
    ]
)

# 保存结果为 CSV
medal_totals_df.to_csv("1_4_02_medal_totals.csv", index=False)
medal_averages_df.to_csv("1_4_02_medal_averages.csv", index=False)

print("已生成文件：medal_totals.csv 和 medal_averages.csv")