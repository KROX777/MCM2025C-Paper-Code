import pandas as pd
import numpy as np

# 输入文件路径
input_txt = "1_1_02_logistic_regression_parameters.txt"
output_csv = "1_1_03_change_dataform.py"

# 默认值
default_params = [-0.01619076, -0.0155581, -0.04892186, -0.14148345211717722]

# 第一行的参数（判断依据）
first_line = [
    0.7446705426356589, 0.9317073170731708, 0.7695675387079551, 0.7548209366391184, 
    0.8377622377622378, 0.7788587956876151, 0.7582781456953642, 0.9286669638876505, 
    1, 0.8001811594202899, 0.8059028787438208, 0.7878787878787878, 0.8369565217391305, 
    0.782608695652174, 0.9148936170212766, 0.9017094017094017, 0.7064220183486238, 
    0.6958368734069669, 0.6822180333019164, 0.9307871267933308, 0.8645833333333334, 
    0.7860125786163522, 0.7748478701825557, 0.7394789579158316, 0.7933333333333333, 
    1, 0.9285714285714286, 0.7371338083927158, 0.8574796960964108, 1, 0.7905737704918033, 
    0.6906885977139671, 0.7835820895522388, 0.8453427065026362, 1, 0.8142857142857143, 
    0.8065755986080091, 0.796423658872077, 0.6607862903225806, 0.8462643678160919, 
    0.7344199424736337, 0.7519025875190258, 0.7795128044971893, 0.7021423635107118, 
    0.8382992969534651, 0.8145549318364074
]

# 读取文本文件
data = []
with open(input_txt, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = eval(line, {"array": np.array})
        data.append(parts)

# 处理数据并写入 CSV
rows = []
for i, (sport, params, last_param) in enumerate(data):
    if first_line[i] < 0.784:
        # 如果参数小于 0.784，使用默认值
        final_params = default_params[:3] + [default_params[3]]
    else:
        # 否则使用原始参数（如果可用）
        if isinstance(params, np.ndarray):
            final_params = params.flatten().tolist() + [last_param]
        else:
            # 如果 params 不是数组，直接使用默认值
            final_params = default_params[:3] + [default_params[3]]
    rows.append([sport] + final_params)

# 创建 DataFrame
columns = ["Sport", "Param1", "Param2", "Param3", "LastParam"]
df = pd.DataFrame(rows, columns=columns)

# 保存为 CSV
df.to_csv(output_csv, index=False)
print(f"转换完成，数据已保存到 {output_csv}")