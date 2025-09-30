#!/bin/bash
# export http_proxy=http://100.68.170.107:3128/
# export https_proxy=http://100.68.170.107:3128/
# export HTTP_PROXY=http://100.68.170.107:3128/
# export HTTPS_PROXY=http://100.68.170.107:3128/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


OUT_DIR=exp/model_Omni_ReflectGRPO
MODEL_NP=/mnt/shared-storage-user/wangjieyi/huoshan/wangjieyi/models/qwen/Qwen/Qwen2___5-Omni-7B # /mnt/shared-storage-user/wangjieyi/huoshan/wangjieyi/ICLR/codes/tryOmni/output/Qwen2___5-Omni-7B-SFT
DATA_FILE=/mnt/shared-storage-user/wangjieyi/huoshan/wangjieyi/ICLR/datasets/AVQA/AVQA/AVQA_dataset/train_r1aqa_line2.json

GPU_NUM=$(nvidia-smi -L | wc -l)
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777

torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --config_path conf/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --out_dir ${OUT_DIR} \
    --data_file ${DATA_FILE} \
    --use_wandb false || exit 1
