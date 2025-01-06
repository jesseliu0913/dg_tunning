#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_baseline.llama32"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=4,6,7 nohup torchrun --master_port=29500 --nproc_per_node=3 tune_combine.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --epoch 2 \
    --task "llama3.2" \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --max_length 2048 > "$LOG_FILE" 2>&1 &
