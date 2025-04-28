#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_combine.qwen"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 tune_combine.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --epoch 2 \
    --task "qwen2.5" \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_length 2048 > "$LOG_FILE" 2>&1 &

