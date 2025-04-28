#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_dialogue.mistral"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=5 nohup torchrun --master_port=29502 --nproc_per_node=1 tune_dialogue.py \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --epoch 2 \
    --task "mistral7b" \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --max_length 1024 > "$LOG_FILE" 2>&1 &
