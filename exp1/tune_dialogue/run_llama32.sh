#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_dialogue.llama32"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=4 nohup torchrun --master_port=29501 --nproc_per_node=1 tune_dialogue.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --epoch 2 \
    --task "llama3.2" \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --max_length 1024 > "$LOG_FILE" 2>&1 &
