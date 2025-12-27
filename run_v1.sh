#!/bin/bash
#./run_v1.sh 1015 7 1
#./run_v1.sh 1015 7 1 &
#nohup ./run_v1.sh 1015 7 1 &
# 检查参数数量是否正确（必须传入3个参数）
if [ $# -ne 3 ]; then
    echo "错误：参数数量不正确"
    echo "使用方式：bash run.sh <test_day> <train_days> <calib_method>"
    echo "示例：bash run.sh 1015 1 1"
    exit 1
fi

# 解析参数
test_day=$1          # 第一个参数：测试日期（如1015）
train_days=$2        # 第二个参数：训练天数（如1）
calib_method=$3      # 第三个参数：校准算法编号（1=hb，2=ir）

# 根据 calib_method 设定 log_pre（hb 或 ir）
case ${calib_method} in
    1)
        log_pre="hb"
        ;;
    2)
        log_pre="ir"
        ;;
    *)
        echo "错误：calib_method 必须为1或2（1=hb，2=ir）"
        exit 1
        ;;
esac

# 定义日志文件路径（包含 log_pre 和 test_day）
log_file="./result/${log_pre}_calibrationV2d3_result_fast_pcvr_flow1_${test_day}.txt"

# 创建result目录（如果不存在）
mkdir -p ./result

# 执行Python脚本，并将输出重定向到日志文件
echo "开始执行校准脚本，参数：test_day=${test_day}, train_days=${train_days}, calib_method=${calib_method}（${log_pre}）"
echo "日志将保存至：${log_file}"

python calib_fast_pcvr_v2d3.py \
    --test_day=${test_day} \
    --train_days=${train_days} \
    --calib_method=${calib_method} \
    > ${log_file} 2>&1  # 2>&1表示将错误输出也重定向到日志文件

# 检查脚本执行结果
if [ $? -eq 0 ]; then
    echo "脚本执行成功，日志路径：${log_file}"
else
    echo "脚本执行失败，详情请查看日志：${log_file}"
    exit 1
fi
