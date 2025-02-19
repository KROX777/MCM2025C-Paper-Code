import pandas as pd
import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

PREDICT_YEAR = args.predict_year

default_values = {'p_max': 0.5, 'a_max': 5, 'alpha': 0.2, 'beta': 0.05}
lambdas = [0.1, 0.3, 0.6, 0.9]

# 读取数据
df = pd.read_csv('1_2_04_average_values_by_noc_sport_final.csv', sep=',', encoding='utf-8-sig')

# 处理 Year 列：除以 4
df['First Year'] = df['First Year'] / 4

# 定义填充缺失年份的函数
def fill_missing_years(group):
    """
    对每个 (Sport, Event, NOC) 组合，填充缺失的年份，并为数值列设置默认值或动态填充。
    """
    # 获取最小和最大年份
    min_year = group['First Year'].min()
    max_year = PREDICT_YEAR / 4
    
    # 生成完整的年份范围
    full_years = np.arange(min_year, max_year + 1)
    
    # 创建一个包含所有年份的 DataFrame
    full_df = pd.DataFrame({'First Year': full_years})
    
    # 合并原始数据
    merged_df = pd.merge(full_df, group, on='First Year', how='left')
    
    # 填充类别列信息（如 Sport, Event, NOC）
    for col in ['Sport', 'NOC']:
        merged_df[col] = merged_df[col].fillna(method='ffill')  # 前向填充
        merged_df[col] = merged_df[col].fillna(method='bfill')  # 后向填充
    
    # 填充数值列
    for i in range(len(merged_df)):
        if pd.isna(merged_df.loc[i, 'p_max']):  # 如果当前行是缺失值
            if i > 0 and not merged_df.loc[i - 1, 'p_max'] == default_values['p_max']:
                # 前一年有值且不是默认值
                merged_df.loc[i, 'p_max'] = merged_df.loc[i - 1, 'p_max'] * 0.9
                merged_df.loc[i, 'a_max'] = merged_df.loc[i - 1, 'a_max']
                merged_df.loc[i, 'alpha'] = merged_df.loc[i - 1, 'alpha']
                merged_df.loc[i, 'beta'] = merged_df.loc[i - 1, 'beta']
            else:
                # 前一年没有值或是默认值，填充默认值
                merged_df.loc[i, 'p_max'] = default_values['p_max']
                merged_df.loc[i, 'a_max'] = default_values['a_max']
                merged_df.loc[i, 'alpha'] = default_values['alpha']
                merged_df.loc[i, 'beta'] = default_values['beta']
    
    return merged_df

# 对每个 (Sport, Event, NOC) 组合填充缺失年份
filled_df = df.groupby(['Sport', 'NOC']).apply(fill_missing_years).reset_index(drop=True)

filled_df.to_csv('1_2_05_average_values_by_noc_sport_filled.csv', index=False)

df = filled_df

# 将填充后的数据赋值回原始 df
def fill_missing_years(group):
    """
    对每个 (Sport, Event, NOC) 组合，填充缺失的年份，并为数值列设置默认值或动态填充。
    """
    # 获取最小和最大年份
    min_year = group['First Year'].min()
    max_year = PREDICT_YEAR / 4
    
    # 生成完整的年份范围
    full_years = np.arange(min_year, max_year + 1)
    
    # 创建一个包含所有年份的 DataFrame
    full_df = pd.DataFrame({'First Year': full_years})
    
    # 合并原始数据
    merged_df = pd.merge(full_df, group, on='First Year', how='left')
    
    # 填充类别列信息（如 Sport, Event, NOC）
    for col in ['Sport', 'NOC']:
        merged_df[col] = merged_df[col].fillna(method='ffill')  # 前向填充
        merged_df[col] = merged_df[col].fillna(method='bfill')  # 后向填充
    
    # 填充数值列
    for i in range(len(merged_df)):
        if pd.isna(merged_df.loc[i, 'p_max']):  # 如果当前行是缺失值
            if i > 0 and not merged_df.loc[i - 1, 'p_max'] == default_values['p_max']:
                # 前一年有值且不是默认值
                merged_df.loc[i, 'p_max'] = merged_df.loc[i - 1, 'p_max'] * 0.9
                merged_df.loc[i, 'a_max'] = merged_df.loc[i - 1, 'a_max']
                merged_df.loc[i, 'alpha'] = merged_df.loc[i - 1, 'alpha']
                merged_df.loc[i, 'beta'] = merged_df.loc[i - 1, 'beta']
            else:
                # 前一年没有值或是默认值，填充默认值
                merged_df.loc[i, 'p_max'] = default_values['p_max']
                merged_df.loc[i, 'a_max'] = default_values['a_max']
                merged_df.loc[i, 'alpha'] = default_values['alpha']
                merged_df.loc[i, 'beta'] = default_values['beta']
    
    return merged_df

# 对每个 (Sport, Event, NOC) 组合填充缺失年份
df = df.groupby(['Sport', 'NOC']).apply(fill_missing_years).reset_index(drop=True)

# 定义 EWMA 预测函数
def predict_with_ewma(series, alpha=0.3, forecast_steps=1):
    """
    使用 EWMA 预测时间序列。
    
    参数:
    - series: 时间序列数据（pandas Series）。
    - alpha: EWMA 的平滑系数，范围 [0, 1]，默认 0.3。
    - forecast_steps: 预测的步数，默认 1。
    
    返回:
    - predicted_values: 预测值列表。
    """
    print("hi")
    if series.empty:  # 如果序列为空，返回默认值
        return [0.0] * forecast_steps  # 返回 forecast_steps 个 0.0
    predicted_values = []
    for step in range(forecast_steps):
        ewma = series.ewm(alpha=alpha, adjust=False).mean()
        last_ewma = ewma.iloc[-1]
        predicted_values.append(last_ewma)
        # 使用 pd.concat 替代 append
        series = pd.concat([series, pd.Series([last_ewma])], ignore_index=True)  # 将预测值加入序列
    return predicted_values

def predict_with_arima(series, order=(1, 1, 1), forecast_steps=1):
    """
    使用 ARIMA 预测时间序列。
    
    参数:
    - series: 时间序列数据（pandas Series）。
    - order: ARIMA 模型的 (p, d, q) 参数，默认 (1, 1, 1)。
    - forecast_steps: 预测的步数，默认 1。
    
    返回:
    - predicted_values: 预测值列表。
    """
    if series.empty or len(series) < 2:  # 如果序列为空或长度不足，返回默认值
        return [0.0] * forecast_steps  # 返回 forecast_steps 个 0.0
    
    try:
        # 拟合 ARIMA 模型
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        # 预测
        predicted_values = model_fit.forecast(steps=forecast_steps)
        return predicted_values.tolist()
    except Exception as e:
        print(f"ARIMA 模型拟合失败: {e}")
        return [0.0] * forecast_steps  # 如果模型拟合失败，返回默认值

# 定义预测函数
def predict_and_update(group, lambdas):
    """
    对每个 (NOC, Sport) 的时间序列进行预测和更新。
    输出所有年份模型值，并对 2008、2012、2016、2020 的模型值应用加权更新。
    """
    years_to_predict = [(PREDICT_YEAR - 16) / 4, (PREDICT_YEAR - 12) / 4, 
                        (PREDICT_YEAR - 8) / 4, (PREDICT_YEAR - 4) / 4]
    updated_values = []

    for year in sorted(group['First Year'].unique()):
        updated_row = {'NOC': group.iloc[0]['NOC'], 'Sport': group.iloc[0]['Sport'], 'First Year': year}
        
        # 检查是否需要进行预测和更新
        if year in years_to_predict:
            lambda_value = lambdas[years_to_predict.index(year)]  # 获取对应年份的 lambda 值
            
            # 过滤出预测年份之前的数据
            filtered_data = group[group['First Year'] < year]
            
            for col in ['p_max', 'a_max', 'alpha', 'beta']:
                # 提取时间序列
                series = filtered_data.set_index('First Year')[col]
                
                # 预测值
                predicted_value = predict_with_ewma(series, alpha=0.3, forecast_steps=1)[0]
                
                # 获取真实值
                actual_value = group[group['First Year'] == year][col]
                actual_value = actual_value.iloc[0] if not actual_value.empty else np.nan
                
                # 更新模型值
                if not np.isnan(actual_value):
                    updated_value = lambda_value * predicted_value + (1 - lambda_value) * actual_value
                else:
                    updated_value = predicted_value  # 如果真实值缺失，直接使用预测值
                
                updated_row[col] = updated_value
        else:
            # 对非目标年份直接保留原始值
            for col in ['p_max', 'a_max', 'alpha', 'beta']:
                original_value = group[group['First Year'] == year][col]
                updated_row[col] = original_value.iloc[0] if not original_value.empty else np.nan
        
        updated_values.append(updated_row)
    
    return pd.DataFrame(updated_values)

# 按 (NOC, Sport) 分组并应用预测和更新逻辑

df = df.groupby(['NOC', 'Sport'], group_keys=False).apply(predict_and_update, lambdas=lambdas)

df['First Year'] = df['First Year'] * 4
# 保存结果到 CSV 文件
df.to_csv('1_2_05_noc_sport_all_years_updated_models.csv', index=False)

print("预测并更新完成，结果已保存到 '05_noc_sport_all_years_updated_models.csv'")

