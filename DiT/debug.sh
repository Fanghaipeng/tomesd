#!/bin/bash

# 显式设置环境变量
export CUDA_VISIBLE_DEVICES=7
torchrun \
    --master_port=16688 --nnodes=1 --nproc_per_node=1 \
    /data1/fanghaipeng/project/sora/tomesd/DiT/sample_ddp_foreach.py \
    --model DiT-XL/2 \
    --num-fid-samples 2000 \
    --sample-dir /data1/fanghaipeng/project/sora/tomesd/DiT/samples \
    --image-size 512 \
    --per-proc-batch-size 32 \
    --tome-type ToMe \
    --ratio 0.05 \
    --ratio-start 0.05 \
    --ratio-end 0.05 \
    --prune-replace \