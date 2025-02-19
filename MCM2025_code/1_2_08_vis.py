import pandas as pd
import argparse
# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

target_year = args.predict_year
df = pd.read_csv('1_2_07_final_results_' + str(target_year) + '_predict.csv')

# 扔掉没有 2024 年数据的行
df = df.dropna(subset=['real_medal'])

# 按 (NOC, Event) 排序
df = df.sort_values(by=['NOC', 'Event'])

# 映射预测结果
def map_result(result):
    if 0 <= result < 3.2:
        return 0
    elif 3.2 <= result < 5:
        return 1
    elif 5 <= result < 10:
        return 2
    else:
        return 3

df['mapped_result'] = df['result'].apply(map_result)

df['real_medal'] = df['real_medal'] / df['Division']
df['mapped_result'] = df['mapped_result'] / df['Division']

# 计算差值
df['difference'] = df['mapped_result'] - df['real_medal']

# 统计所有差值的和
total_difference = df['difference'].sum()

equal_count = (df['mapped_result'] == df['real_medal']).sum()
total_count = len(df)
equal_probability = equal_count / total_count

# 保存结果到新的 CSV 文件
df.to_csv('1_2_08_final_results_with_differences.csv', index=False)

print("处理完成！结果已保存为 'final_results_with_differences.csv'")
print(f"所有差值的总和为: {total_difference}")

print(f"`mapped_result` 和 `real_medal` 相等的概率为: {equal_probability:.2%}")