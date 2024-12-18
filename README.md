# dg_tunning


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tune_umls.py> ./log/mc.log 2>&1 &
