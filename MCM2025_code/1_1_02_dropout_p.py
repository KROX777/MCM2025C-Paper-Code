import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict

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
                label = int(parts[2])
                data[sport].append((medal, t, label))

    return data

# 读取数据
file_path = '1_1_01_output_sep.txt'  # 替换为你的文件路径
sport_data = read_data(file_path)

def MapRank(medal, t):
    if medal == 'Gold':
        return [0.9, 0.8, np.exp(t)]
    elif medal == 'Silver':
        return [0.7, 0.5, np.exp(t)]
    elif medal == 'Bronze':
        return [0.4, 0.3, np.exp(t)]
    else:
        return [0.1, 0.1, np.exp(t)]

# -------------------------------Data Preprocessing---------------------------------------

# 假设你的数据是 m_last, t, labels
# 将 m_last 和 t 组合成特征矩阵 X

# -----------------------------------------------------------------------------------------------

'''
solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},
      default: ‘liblinear’ Algorithm to use in the optimization problem.
$  Small datasets:                 ‘liblinear’ 
   Large datasets:                 ‘sag’ , ‘saga’ 
$  Multiclass problems:             ‘newton-cg’, ‘sag’, ‘saga’ , ‘lbfgs’ 
   One-versus-rest:                ‘liblinear’ 
$  L2 penalty:                     ‘newton-cg’, ‘lbfgs’ , ‘sag’
   L1 penalty:                     ‘liblinear’ , ‘saga’
'''

score_all = []
res = []

for sport, data in sport_data.items():
    print(f"Training model for Sport: {sport}")

    # 解析数据
    medal, t, labels = zip(*data)
    m_last_and_t = [MapRank(m, ti) for m, ti in zip(medal, t)]

# 堆叠特征向量
    X = np.row_stack(m_last_and_t)
    y = np.array(labels)  # 标签

    # 划分训练集和测试集
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

    # 打印数据形状
    print("训练集特征形状:", trainX.shape)
    print("训练集标签形状:", trainy.shape)
    print("测试集特征形状:", testX.shape)
    print("测试集标签形状:", testy.shape)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        score_all.append(1)
        res.append((sport, 0, 0))
        continue


    # ----------------------------------- Training -------------------------------------------
    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class='multinomial').fit(trainX, trainy)
    # --------------------------------------------------------------------------------------------

    predict = clf.predict(testX)
    coefficients = clf.coef_  # 权重系数
    intercept = clf.intercept_  # 截距
    res.append((sport, coefficients, intercept[0]))
    ts = clf.score(trainX, trainy)
    score_all.append(ts)

    # print the training scores
    print("predict precision:", sum(predict == testy) / len(testy))
    print("training score : %.3f" % ts)

    # -----------------------------------Visualization----------------------------------------
    # create a mesh to plot in
    # if len(clf.classes_) == 1:
    #     print(f"Only one class in {sport}, skipping visualization.")
    #     continue
    # h = .02  # step size in the mesh
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))

    # # Plot the decision boundary. For that, we will assign a color to each
    # # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # plt.title(f"Decision surface of LogisticRegression ({sport})")
    # plt.axis('tight')

    # # Plot also the training points
    # colors = "bry"
    # for i, color in zip(clf.classes_, colors):
    #     idx = np.where(y == i)
    #     plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired,
    #                 edgecolor='black', s=20)

    # # Plot the three one-against-all classifiers
    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    # coef = clf.coef_
    # intercept = clf.intercept_

    # def plot_hyperplane(c, color):
    #     def line(x0):
    #         return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    #     plt.plot([xmin, xmax], [line(xmin), line(xmax)],
    #              ls="--", color=color)

    # for i, color in zip(clf.classes_, colors):
    #     plot_hyperplane(i, color)
    # --------------------------------------------------------------------------------------------------

import seaborn as sns

sns.set(style="whitegrid")  # 使用白色网格背景
sns.set_palette("pastel")   # 使用柔和的调色板

# 创建箱线图
plt.figure(figsize=(8, 6))  # 设置图表大小
ax = sns.boxplot(x=score_all, width=0.5, linewidth=2.5)

# 添加标题和标签
plt.xlabel("Accuracy Score", fontsize=12)
plt.title("Boxplot of Accuracy Scores", fontsize=14, fontweight='bold')

# 美化坐标轴
ax.tick_params(axis='both', which='major', labelsize=10)

# 保存图表
output_file = '1_1_02_boxplot_accuracy_scores.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
print(f"Boxplot has been saved to {output_file}")

# 显示图表
plt.show()

output_file = '1_1_02_logistic_regression_parameters.txt'
with open(output_file, 'w') as f:
    f.write(str(score_all)+"\n")
    for i in res:
        f.write(str(i)+"\n")