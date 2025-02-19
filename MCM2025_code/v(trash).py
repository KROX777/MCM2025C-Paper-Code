import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import defaultdict

# === 1. 数据读取与解析 ===
def read_data(file_path):
    data = defaultdict(list)  # 按Sport分组存储数据

    with open(file_path, 'r') as f:
        lines = f.readlines()
        sport = None
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 1 or parts[1].isalpha():  # 判断是否为Sport名称
                print(sport)
                sport = line.strip()
            else:
                medal = parts[0]
                t = float(parts[1])
                label = float(parts[2])
                data[sport].append((medal, t, label))

    return data

# 读取数据
file_path = 'output.txt'  # 替换为你的文件路径
sport_data = read_data(file_path)

# === 2. 定义逻辑回归模型 ===
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1))  # 初始化 alpha
        self.beta = nn.Parameter(torch.randn(1))   # 初始化 beta
        self.lambd = nn.Parameter(torch.randn(1))  # 初始化 lambda

    def forward(self, m_last, t):
        linear = self.alpha * m_last - self.beta * torch.exp(self.lambd * t)
        return torch.sigmoid(linear)  # 返回概率

# === 3. 训练每个Sport的模型 ===
results = {}  # 保存每个Sport的结果

def MapRank(medal):
    if medal == 'Gold':
        return 10
    elif medal == 'Silver':
        return 8
    elif medal == 'Bronze':
        return 6
    else:
        return 1

for sport, data in sport_data.items():
    print(f"Training model for Sport: {sport}")

    # 解析数据
    medal, t, labels = zip(*data)
    m_last = MapRank(medal)
    m_last = torch.tensor(m_last, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # 初始化模型和优化器
    model = LogisticRegressionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()  # 二分类交叉熵损失

    # 模型训练
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(m_last, t)  # 预测概率
        loss = loss_fn(predictions, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 保存学到的参数
    results[sport] = {
        'alpha': model.alpha.item(),
        'beta': model.beta.item(),
        'lambda': model.lambd.item(),
    }

    # 绘制 ROC 曲线
    predictions = model(m_last, t).detach().numpy()
    fpr, tpr, _ = roc_curve(labels.numpy(), predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {sport}")
    plt.legend()
    plt.savefig(f"roc_curve_{sport}.png")  # 保存ROC曲线
    plt.close()

# === 4. 将每个Sport的结果输出到文件 ===
output_file = 'sport_results.txt'  # 输出文件名
with open(output_file, 'w') as f:
    f.write("Learned Parameters for Each Sport:\n")
    for sport, params in results.items():
        f.write(f"Sport: {sport}\n")
        f.write(f"Alpha: {params['alpha']:.4f}\n")
        f.write(f"Beta: {params['beta']:.4f}\n")
        f.write(f"Lambda: {params['lambda']:.4f}\n")
        f.write("\n")