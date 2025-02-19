import subprocess
import csv
import numpy as np

# 超参数范围
a_values = np.arange(0, 0.4 + 0.005, 0.04)  # a 的范围 [0, 0.4]，步长 0.005
b_values = np.arange(0, 0.2 + 0.005, 0.04)  # b 的范围 [0, 0.2]，步长 0.005

# 输出文件名
output_csv = "results.csv"

# 存储结果
results = []

# 遍历所有超参数组合
for a in a_values:
    for b in b_values:
        print(a,b)
        try:
            # 调用 1_2_02_gamma_get.py 并传递参数 a 和 b
            process = subprocess.run(
                ['python', '1_2_02_gamma_get.py', str(a), str(b)],
                capture_output=True,
                text=True,
                check=True
            )
            # 获取输出结果
            output = process.stdout.strip()
            # 添加到结果列表
            results.append((a, b, output))
        except subprocess.CalledProcessError as e:
            # 如果发生错误，将错误信息存储
            results.append((a, b, f"Error: {e.stderr.strip()}"))

# 将结果写入 CSV 文件
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入标题
    csv_writer.writerow(["a", "b", "output"])
    # 写入所有数据
    csv_writer.writerows(results)

print(f"结果已保存到 {output_csv}")