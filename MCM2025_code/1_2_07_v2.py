import pandas as pd
import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="Filter data by year")
parser.add_argument("--predict-year", type=int, required=True, help="Year to filter data")
args = parser.parse_args()

target_year = args.predict_year
ini_age = 16

# 加载数据
summer_oly_df = pd.read_csv('summerOly_athletes.csv')
fitted_results_df = pd.read_csv('1_2_03_athlete_event_fitted_results_updated.csv')
noc_sport_models_df = pd.read_csv('1_2_05_noc_sport_all_years_updated_models.csv')

# 提取 Year=2020 的 (Name, Sport, Event)

target_athletes = summer_oly_df[summer_oly_df['Year'] == target_year - 4][['Name', 'Sport', 'Event']]

target_athletes = target_athletes.merge(
    fitted_results_df, on=['Name', 'Sport', 'Event'], how='left'
)

# 从 noc_sport_models_df 获取 First Year 对应的 p_c, a_c, al_c, b_c
target_athletes = target_athletes.merge(
    noc_sport_models_df, left_on=['NOC', 'Sport', 'First Year'],
    right_on=['NOC', 'Sport', 'First Year'], how='left',
    suffixes=('', '_c')
)

with open('1_2_06_gamma_p(s).txt', 'r') as f:
    gamma_values = f.read().split()
p_g, a_g, al_g, b_g = map(float, gamma_values)  # 转换为浮点数



# 计算 t = 2024 - First Year
target_athletes['t'] = target_year - target_athletes['First Year'] + ini_age

# 计算最终结果
def calculate_result(row):
    t = row['t']
    
    # 第一项：p_max * (t / a_max)**alpha * np.exp(-beta * (t - a_max))
    term1 = row['p_max'] * (t / row['a_max'])**row['alpha'] * np.exp(-row['beta'] * (t - row['a_max']))
    
    # 第二项：p_c * (t / a_c)**al_c * np.exp(-b_c * (t - a_c))
    term2 = row['p_max_c'] * (t / row['a_max_c'])**row['alpha_c'] * np.exp(-row['beta_c'] * (t - row['a_max_c']))
    
    # 第三项：p_g * (t / a_g)**al_g * np.exp(-b_g * (t - a_g))
    term3 = p_g * (t / a_g)**al_g * np.exp(-b_g * (t - a_g))
    
    # 最终结果
    return 0.5 * term1 + 0.5 * (term1 * term2) / term3

target_athletes['result'] = target_athletes.apply(calculate_result, axis=1)

def get_real_medal(row, summer_oly_df):
    medal = summer_oly_df[
        (summer_oly_df['Name'] == row['Name']) &
        (summer_oly_df['Sport'] == row['Sport']) &
        (summer_oly_df['Event'] == row['Event']) &
        (summer_oly_df['Year'] == target_year)
    ]['Medal'].values
    if len(medal) > 0:
        # 将奖牌转换为数值
        medal_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1, 'No medal': 0}
        return medal_map.get(medal[0], 0)  # 如果没有匹配的奖牌，默认为 0
    else:
        return np.nan  # 如果没有 2024 年的记录，返回 NaN

target_athletes['real_medal'] = target_athletes.apply(
    lambda row: get_real_medal(row, summer_oly_df), axis=1
)

# 保存结果到新的 CSV 文件
target_athletes.to_csv('1_2_07_final_results_' + str(target_year) + '_predict.csv', index=False)

print("处理完成！结果已保存为 'final_results_{target_year}_predict.csv'")