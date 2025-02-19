import pandas as pd
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

PREDICT_YEAR = args.predict_year

ALPHA = 0.3  # EWMA 的平滑系数

# 读取数据
data = pd.read_csv("1_3_01_newcomers.csv")
data_gt = pd.read_csv("1_3_01_newcomers_gt.csv")

# 提取年份列（排除非年份列）
non_year_columns = ["NOC", "Sport", "Event", "Division"]
year_columns = [col for col in data.columns if col not in non_year_columns]

# 填充缺失值
def fill_missing_values(row):
    # 只处理年份列
    filled_values = []
    for i, year in enumerate(year_columns):
        value = row[year]
        if pd.notna(value):  # 如果不是缺失值，保留原值
            filled_values.append(value)
        else:  # 如果是缺失值
            if i > 0 and pd.notna(filled_values[i - 1]):  # 如果前一格有值
                filled_values.append(filled_values[i - 1] * 0.9)
            else:  # 如果前一格也没值或是开头
                filled_values.append(0)
    # 返回填充后的值作为新行
    for i, year in enumerate(year_columns):
        row[year] = filled_values[i]
    return row

# 应用缺失值填充函数
data = data.apply(fill_missing_values, axis=1)

# 计算预测值
def predict_with_ewma(row, alpha=ALPHA):
    # 仅使用预测年份及之前的数据
    train_years = [str(year) for year in year_columns if int(year) <= PREDICT_YEAR]
    train_data = row[train_years].dropna()
    # 使用 EWMA 计算预测值
    ewma = train_data.ewm(alpha=alpha).mean().iloc[-1]
    return ewma

data["Predicted"] = data.apply(predict_with_ewma, axis=1)

# 更新 Deviation 列
def calculate_deviation(row):
    # 根据 NOC 和 Sport 从 data_gt 中找到对应的行
    gt_row = data_gt[
        (data_gt["NOC"] == row["NOC"]) & (data_gt["Sport"] == row["Sport"])
    ]
    if gt_row.empty:  # 如果没有找到对应的行，返回 NaN
        return None
    # 提取预测年份对应的值
    gt_value = gt_row.get(str(PREDICT_YEAR))
    if gt_value is not None and not pd.isna(gt_value.values[0]):
        # 如果 data_gt 对应年份有值，计算离差
        return abs(row["Predicted"] - gt_value.values[0]) / row["Division"]
    return None

# 计算每行的离差值
if PREDICT_YEAR <= 2024:
    data["Deviation"] = data.apply(calculate_deviation, axis=1)

# 保存结果到新文件
output_file = "1_3_02_added_number.csv"
data.to_csv(output_file, index=False)

print(f"预测结果已保存到 {output_file}")