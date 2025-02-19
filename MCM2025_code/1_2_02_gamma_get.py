import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys

# 奖牌对应的实力值
medal_scores = {"Gold": 10, "Silver": 6, "Bronze": 4, "No medal": 1}
    # 正则化权重
lambda_alpha = float(sys.argv[1])  # alpha 的正则化权重（很大）
lambda_beta = float(sys.argv[2])   # beta 的正则化权重（很大）
lambda_a_max = 0.0001   # a_max 的正则化权重（一点点）
lambda_p_max = 0.01   # p_max 的正则化权重（不加）

ini_age = 16

BOUND = [(1, None), (ini_age - 12,ini_age + 12), (0.05, 1), (0.05, None)]

ini_p_delta = 1.5

# 加载数据
df = pd.read_csv('1_2_01_athlete_event_medal_year_sorted.csv')
df['Strength'] = df['Medal'].map(medal_scores)

# 为每个 (Name, Sport, Event) 计算赛龄
df['First Year'] = df.groupby(['Name', 'Sport', 'Event'])['Year'].transform('min')
df['Age'] = df['Year'] - df['First Year'] + ini_age

# 定义拟合函数
def performance_model(a, p_max, a_max, alpha, beta):
    """实力模型公式"""
    return p_max * (a / a_max)**alpha * np.exp(-beta * (a - a_max))

# 定义方程



# 定义带权重正则化的损失函数
def weighted_regularized_loss(params, x, y):
    """
    自定义损失函数，加入带权重的正则化。
    
    参数:
    - params: 模型参数 [p_max, a_max, alpha, beta]。
    - x: 自变量（年龄）。
    - y: 因变量（实力值）。
    
    返回:
    - 损失值（均方误差 + 带权重的正则化项）。
    """
    p_max, a_max, alpha, beta = params
    y_pred = performance_model(x, p_max, a_max, alpha, beta)
    mse = np.mean((y - y_pred)**2)  # 均方误差
    

    
    # 正则化项
    regularization = (
        lambda_alpha * np.exp(alpha) +  # alpha 的正则化
        lambda_beta * np.exp(beta) +    # beta 的正则化
        lambda_a_max * a_max**2 +   # a_max 的正则化
        lambda_p_max * p_max**2
    )
    
    return mse + regularization

# 遍历每个 (Name, Sport, Event) 分组，拟合模型
group_fit_losses = []  # 用于计算平均 loss
results = []  # 用于存储所有的分组结果

for (name, sport, event, division), group in df.groupby(['Name', 'Sport', 'Event', 'Division']):
    # print(f"Fitting model for {name} ({sport} - {event})")
    group = group.sort_values(by='Age')
    ages = group['Age'].values
    strengths = group['Strength'].values

    if len(group) == 1:
        # 只有一条记录时直接赋值
        p_max = strengths[0] + ini_p_delta
        alpha = np.random.uniform(0.05, 0.1)  # 在小范围内随机生成 alpha
        beta = np.random.uniform(0.05, 0.5)  # 在小范围内随机生成 beta
        def equation(x):
            return (-alpha * np.log(x) + beta * x - (alpha * np.log(ini_age) - beta * ini_age + np.log(strengths[0] / p_max))) ** 2
        # 使用数值方法求解
        result = minimize_scalar(equation, bounds=(4, 16), method='bounded')
        if result.success:
            a_max = result.x
        else:
            a_max = np.random.uniform(4, 20)
            # print("未找到合适的解")
        fit_loss = 0

        # 保存结果到最后一行
        group['p_max'] = p_max
        group['a_max'] = a_max
        group['alpha'] = alpha
        group['beta'] = beta
        group['fit_loss'] = fit_loss
        results.append(group)
        continue

    # 初始参数猜测
    initial_guess = [max(strengths), np.median(ages), 1, 0.1]
    
    try:
        # 使用 minimize 拟合模型
        res = minimize(
            weighted_regularized_loss, initial_guess,
            args=(ages, strengths),
            bounds=BOUND  # 参数范围
        )
        p_max, a_max, alpha, beta = res.x

        # 计算拟合的实力值和 loss（均方误差）
        fitted_strengths = performance_model(ages, p_max, a_max, alpha, beta)
        fit_loss = np.mean((strengths - fitted_strengths)**2)

        # 保存 loss 和参数到分组的最后一行
        group['p_max'] = np.nan
        group['a_max'] = np.nan
        group['fit_loss'] = np.nan
        group.loc[group.index[-1], 'p_max'] = p_max
        group.loc[group.index[-1], 'a_max'] = a_max
        group.loc[group.index[-1], 'alpha'] = alpha
        group.loc[group.index[-1], 'beta'] = beta
        group.loc[group.index[-1], 'fit_loss'] = fit_loss

        # 保存 loss 供后续计算平均值
        group_fit_losses.append(fit_loss)

        # 保存结果
        results.append(group)

    except Exception as e:
        print(f"Could not fit model for {name} ({sport} - {event}): {e}")
        group['p_max'] = np.nan
        group['a_max'] = np.nan
        group['alpha'] = np.nan
        group['beta'] = np.nan
        group['fit_loss'] = np.nan
        results.append(group)

# 合并所有分组结果
final_df = pd.concat(results)

# 添加总体平均 fit_loss 到数据框最后一行
average_fit_loss = np.mean(group_fit_losses)
final_df.loc[len(final_df)] = {
    'Name': 'Average',
    'Sport': np.nan,
    'Event': np.nan,
    'Year': np.nan,
    'Medal': np.nan,
    'Strength': np.nan,
    'First Year': np.nan,
    'Age': np.nan,
    'p_max': np.nan,
    'a_max': np.nan,
    'fit_loss': average_fit_loss,
}

# 保存为 CSV
final_df.to_csv('1_2_02_athlete_event_fitted_results.csv', index=False)

print(average_fit_loss)