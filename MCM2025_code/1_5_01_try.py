import subprocess
import pandas as pd
import random
import os

# 超参数搜索范围
BASE_PARAMS = {'a': 2.628, 'b': 4.479, 'c': 5.4}
SEARCH_RANGE = 0  # 超参数调整范围
NOC_LIST = ["USA", "CHN", "RUS", "GER", "JPN", "FRA", "KOR", "ITA", "AUS", "NED", "HUN", "BRA", "ESP", "KEN"]

# 定义目标函数
def calculate_v(aggregated_file, Medals_file, noc_list):
    df1 = pd.read_csv(aggregated_file)
    df2 = pd.read_csv(Medals_file)

    # 仅保留在 NOC 列表中的数据
    df1 = df1[df1['NOC'].isin(noc_list)]
    df2 = df2[df2['NOC'].isin(noc_list)]

    # 合并数据集
    merged = pd.merge(df1[['NOC', 'Medal']], df2[['NOC', 'Medal']], on='NOC', suffixes=('_1', '_2'))

    # 计算差的平方和
    merged['Medal_diff_squared'] = (merged['Medal_1'] - merged['Medal_2']) ** 2
    return merged['Medal_diff_squared'].sum()

# 搜索最佳超参数
best_v = float('inf')
parameters = []

for _ in range(1):
    # 随机生成超参数
    a = BASE_PARAMS['a']
    b = BASE_PARAMS['b'] 
    c = BASE_PARAMS['c'] + random.uniform(-SEARCH_RANGE, SEARCH_RANGE)

    # 运行脚本并传递超参数
    subprocess.run(['python3', '1_4_01_getv.py', '--a', str(a), '--b', str(b), '--c', str(c)], check=True)

    # 文件路径
    aggregated_file = '1_4_01_aggregated_by_noc.csv'
    Medals_file = 'old_athletes_Medals_2012_sorted.csv'

    # 确保文件存在
    if not os.path.exists(aggregated_file) or not os.path.exists(Medals_file):
        raise FileNotFoundError("One or more required files are missing after script execution.")

    # 计算目标函数 v
    v = calculate_v(aggregated_file, Medals_file, NOC_LIST) / 14

    # 更新最优超参数
    if v < best_v:
        best_v = v
        parameters = [a, b, c]
        print(f"New best_v found: {best_v}")
    
        

print(f"Minimal v: {best_v}, parameters: {parameters}")