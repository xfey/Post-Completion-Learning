#!/bin/bash

# 检查是否提供了config参数
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 recipes/Qwen2.5-7B-Instruct/rft/debug.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# 检查config文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# 激活虚拟环境
source openr1/bin/activate

# 运行训练命令
accelerate launch --main_process_port=23498 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config "$CONFIG_FILE" \