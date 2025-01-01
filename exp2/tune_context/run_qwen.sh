#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_context.qwen"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --master_port=29501 --nproc_per_node=2 tune_context.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --epoch 2 \
    --task "qwen2.5" \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_length 1024 > "$LOG_FILE" 2>&1 &

