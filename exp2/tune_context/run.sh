#!/bin/bash
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tune_context.py \

> ./log/mc.log 2>&1 &
