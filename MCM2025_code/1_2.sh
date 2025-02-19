#!/bin/bash

PREDICT_YEAR=1996

scripts=(
    "1_2_01_gamma_preparation.py"
    "1_2_02_gamma_get.py"
    "1_2_03_pc_pre.py"
    "1_2_04_pc_pre2.py"
    "1_2_05_ewma.py"
    "1_2_06_gamma_ps.py"
    "1_2_07_v2.py"
    "1_2_08_vis.py"
)

# 遍历并依次执行脚本
for script in "${scripts[@]}"; do
    echo "正在执行 $script ..."
    python3 "$script" --predict-year $PREDICT_YEAR  # 使用 python3 运行脚本
    if [ $? -ne 0 ]; then  # 检查脚本是否成功执行
        echo "$script 执行失败，停止运行。"
        exit 1
    fi
    echo "$script 执行完成。"
done

echo "所有脚本执行完毕！"