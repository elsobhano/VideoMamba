#!/bin/bash
JOB_NAME='videomamba_tiny'

OUTPUT_DIR="./${JOB_NAME}"
LOG_DIR="./${JOB_NAME}"
PREFIX='/home/sobhan/Documents/Datasets/MASKED_VIDEOS/'
DATA_PATH='/home/sobhan/Documents/Code/VideoMamba/videomamba/video_sm/datasets/data'
PRETRAINED_PATH='/home/sobhan/Documents/Code/VideoMamba/videomamba_s16_k400_f16_res224.pth'
export PYTHONPATH=.

cd "causal-conv1d"
pip install -e .
cd ..
cd "mamba"
pip install -e .
cd ..
cd "videomamba/video_sm"

python run_class_finetuning.py \
    --model videomamba_small \
    --finetune ${PRETRAINED_PATH} \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'Meign-V' \
    --split ',' \
    --nb_classes 2301 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --orig_t_size 16 \
    --num_workers 12 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 70 \
    --lr 4e-4 \
    --drop_path 0.15 \
    --aa rand-m5-n2-mstd0.25-inc1 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
    --no_auto_resume \
    --bf16
