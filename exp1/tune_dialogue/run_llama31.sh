#!/bin/bash
export NCCL_P2P_DISABLE=1

LOG_FOLDER="./logs"
LOG_FILE="$LOG_FOLDER/tune_dialogue.llama31"

if [ ! -d "$LOG_FOLDER" ]; then
    echo "Log folder does not exist. Creating one..."
    mkdir -p "$LOG_FOLDER"
else
    echo "Log folder exists."
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 tune_umls_dialogue.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --epoch 3 \
    --task "llama3.1" \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --max_length 1024 > "$LOG_FILE" 2>&1 &
