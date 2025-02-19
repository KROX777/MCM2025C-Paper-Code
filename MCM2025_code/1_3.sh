#!/bin/bash

PREDICT_YEAR=2028

# 列出要执行的 Python 脚本
scripts=(
    "1_3_01_get_added_numbers.py"
    "1_3_02_predict_added_numbers.py"
    "1_4_01_getv.py"
    "1_4_02_getr.py"
    "1_4_03_getfinal.py"
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