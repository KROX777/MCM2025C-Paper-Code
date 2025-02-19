import pandas as pd
import numpy as np
import argparse

# 加载数据
# parser = argparse.ArgumentParser(description="Filter data by year")
# parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
# args = parser.parse_args()

parser = argparse.ArgumentParser(description="Process hyperparameters for 1_4_01_getv.py")
parser.add_argument('--a', type=float, required=True, help="Hyperparameter a")
parser.add_argument('--b', type=float, required=True, help="Hyperparameter b")
parser.add_argument('--c', type=float, required=True, help="Hyperparameter c")

args = parser.parse_args()

import random

def random_with_probability(p):
    """
    根据概率 p 返回 1 或 0
    :param p: float, 返回 1 的概率 (0 <= p <= 1)
    :return: int, 1 或 0
    """
    return 1 if random.random() < p else 0

target_year = 2016
differences_csv = '1_2_07_final_results_' + str(target_year) + '_predict.csv'
parameters_csv = "1_1_03_sports_parameters.csv"
output_detailed_csv = "updated_results_with_p.csv"
output_sport_csv = "1_4_01_aggregated_by_sport.csv"
output_noc_csv = "1_4_01_aggregated_by_noc.csv"

a = args.a
b = args.b
c = args.c

usa_gold_boost = 0.3
chn_gold_decay = 0.6

def map_result(result):
    if 0 <= result < a:
        return 0
    elif a <= result < b:
        return 1
    elif b <= result < c:
        return 2
    else:
        return 3

# 加载数据
differences_df = pd.read_csv(differences_csv)
parameters_df = pd.read_csv(parameters_csv)

def MapRank(medal, t):
    if medal == 'Gold':
        return [0.9, 0.8, np.exp(t)]
    elif medal == 'Silver':
        return [0.7, 0.5, np.exp(t)]
    elif medal == 'Bronze':
        return [0.4, 0.3, np.exp(t)]
    else:
        return [0.1, 0.1, np.exp(t)]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义计算 p 的函数（您可以自行实现）
def calculate_p(param1, param2, param3, last_param, medal, t):
    # 自定义计算公式
    z = np.dot(MapRank(medal, t), [param1, param2, param3]) + last_param
    # 计算概率
    probabilities = sigmoid(z)
    return probabilities

# 添加 p 值和分配奖牌概率
results = []
for _, row in differences_df.iterrows():
    name, noc, sport, event = row["Name"], row["NOC"], row["Sport"], row["Event"]
    medal = row["Medal"]
    mapped_medal = map_result(row["result"])
    t = int(row["Year"] - row["First Year"]) / 4 + 1
    
    # 获取对应 Sport 的参数
    sport_params = parameters_df[parameters_df["Sport"] == sport]
    if sport_params.empty:
        continue  # 如果没有找到对应的 Sport 参数，跳过

    param1, param2, param3, last_param = sport_params.iloc[0, 1:]  # 获取参数
    p = calculate_p(param1, param2, param3, last_param, medal, t)

    # 根据 mapped_medal 分配概率
    gold = silver = bronze = no_medal = 0
    # if row["NOC"] == "CHN" and mapped_medal == 3:
    #     p *= chn_gold_decay
    if mapped_medal == 3:
        gold = p / row["Division"]
    elif mapped_medal == 2:
        silver = p / row["Division"]
    elif mapped_medal == 1:
        bronze = p / row["Division"]
    else:
        no_medal = p / row["Division"]

    results.append({
        "Name": name,
        "NOC": noc,
        "Sport": sport,
        "Event": event,
        "Gold": gold,
        "Silver": silver,
        "Bronze": bronze,
        "NoMedal": no_medal
    })



# 创建详细结果 DataFrame
detailed_df = pd.DataFrame(results)
detailed_df.to_csv(output_detailed_csv, index=False)
print(f"详细结果已保存到 {output_detailed_csv}")

# 按 (NOC, Sport, Event) 聚合
print(detailed_df.columns)
sport_agg = detailed_df.groupby(["NOC", "Sport", "Event"]).sum(numeric_only=True).reset_index()
sport_agg.to_csv(output_sport_csv, index=False)
print(f"按 (NOC, Sport, Event) 聚合的结果已保存到 {output_sport_csv}")

# 按 NOC 聚合
noc_agg = sport_agg.groupby("NOC").sum(numeric_only=True).reset_index()
noc_agg["Medal"] = noc_agg["Gold"] + noc_agg["Silver"] + noc_agg["Bronze"]
noc_agg = noc_agg.sort_values(
    by=["Gold", "Silver", "Bronze"], 
    ascending=[False, False, False]
)
noc_agg.to_csv(output_noc_csv, index=False)
print(f"按 NOC 聚合的结果已保存到 {output_noc_csv}")