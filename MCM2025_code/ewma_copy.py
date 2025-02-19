import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # 导入正确的 ARIMA 模块

# 读取数据
df = pd.read_csv('athlete_event_noc_sorted_filtered.csv', sep=',', encoding='utf-8-sig')

# 处理 Year 列：除以 4
df['First Year'] = df['First Year'] / 4

# 定义填充缺失年份的函数
def fill_missing_years(group):
    """
    对每个 (Sport, Event, NOC) 组合，填充缺失的年份，并为数值列设置默认值。
    """
    # 获取最小和最大年份
    min_year = 1952/4
    max_year = 2024/4
    
    # 生成完整的年份范围
    full_years = np.arange(min_year, max_year + 1)
    
    # 创建一个包含所有年份的 DataFrame
    full_df = pd.DataFrame({'First Year': full_years})
    
    # 合并原始数据
    merged_df = pd.merge(full_df, group, on='First Year', how='left')
    
    # 填充其他列的信息（如 Sport, Event, NOC）
    for col in ['Sport', 'Event', 'NOC']:
        merged_df[col] = merged_df[col].fillna(method='ffill')  # 前向填充
        merged_df[col] = merged_df[col].fillna(method='bfill')  # 后向填充
    
    # 设置默认值
    default_values = {'p_max': 0.5, 'a_max': 8, 'alpha': 1, 'beta': 0.2}
    for col, default_value in default_values.items():
        merged_df[col] = merged_df[col].fillna(default_value)
    
    return merged_df

# 对每个 (Sport, Event, NOC) 组合填充缺失年份
filled_df = df.groupby(['Sport', 'Event', 'NOC']).apply(fill_missing_years).reset_index(drop=True)

# 检查填充后的数据
print("填充后的数据：")
print(filled_df.head())

# 将填充后的数据赋值回原始 df
df = filled_df

# 对每个 (Sport, Event, NOC) 组合填充缺失年份
df = df.groupby(['Sport', 'Event', 'NOC']).apply(fill_missing_years).reset_index(drop=True)
print(df)

# 定义 EWMA 预测函数
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
def predict_future_years(group):
    """
    对每个 (Sport, Event, NOC) 组合预测 2016、2020、2024、2028 年的数据。
    同时输出真实值和差值的平方。
    """
    years_to_predict = [2016, 2020, 2024, 2028]
    predictions = {}
    print("hi")
    for year in years_to_predict:
        # 过滤掉预测年份及以后的数据
        filtered_data = group[group['First Year'] < year / 4]
        
        # 对每个数值列进行预测
        for col in ['p_max', 'a_max', 'alpha', 'beta']:
            # 提取时间序列
            series = filtered_data.set_index('First Year')[col]
            
            # 预测
            predicted_value = predict_with_arima(series, order=(1, 1, 1), forecast_steps=1)[0]
            
            # 获取真实值（如果存在）
            actual_value = group[group['First Year'] == year / 4][col]
            actual_value = actual_value.iloc[0] if not actual_value.empty else np.nan
            
            # 计算差值的平方
            if not np.isnan(actual_value):
                squared_error = (predicted_value - actual_value) ** 2
            else:
                squared_error = np.nan
            
            # 存储结果
            predictions[f'{year}_{col}_predicted'] = predicted_value
            predictions[f'{year}_{col}_actual'] = actual_value
            predictions[f'{year}_{col}_squared_error'] = squared_error
    
    return pd.Series(predictions)


# 对每个 (Sport, Event, NOC) 组合进行预测
predictions = df.groupby(['Sport', 'Event', 'NOC']).apply(predict_future_years).reset_index()


output_file = 'p(c)_predictions_new_arima.csv'
predictions.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"预测结果已写入文件: {output_file}")