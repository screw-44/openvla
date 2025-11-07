#!/usr/bin/env bash
set -euo pipefail

# Stop Warnings（这个是莫名其妙什么奇怪的高层库里面的warninig，贼逆天）
export PYTHONWARNINGS="ignore::UserWarning:pydantic._internal._generate_schema,ignore:UnsupportedFieldAttributeWarning"
# '2' 会屏蔽 INFO 和 WARNING 级别的日志，能有效清理大部分噪音
export TF_CPP_MIN_LOG_LEVEL=2
# 3. torch警告
export OMP_NUM_THREADS=1
# cuda warning
export TF_CPP_MIN_LOG_LEVEL=3

# Paths
# Resolve script directory and use dataset relative to the script location so the
# script works when run from the repo (avoids hard-coded external absolute paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT_DIR="${SCRIPT_DIR}/dataset/ob_hand_hold_object_frame_rlds"  # RLDS root of your custom dataset
RUN_ROOT_DIR="runs"                                                                               # Where logs/checkpoints go

# Tracking (optional; defaults exist in train.py)
WANDB_PROJECT="openvla"
WANDB_ENTITY="sdafasj1231"

# Save cadence (reflecting cus_train.py PoC setting)
SAVE_INTERVAL=100

# If you want to resume from a checkpoint, uncomment and set these:
# PRETRAINED_CKPT="/absolute/path/to/checkpoint.pt"
IS_RESUME=true
PRETRAINED_CKPT="/inspire/ssd/project/robot-decision/hexinyu-253108100063/openvla/runs/siglip-224px+custom-trajectory+n0+b64+x7_object_frame/checkpoints/step-000200-epoch-50-loss=0.0223.pt"
RESUME_STEP=200
RESUME_EPOCH=50

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "siglip-224px+custom-trajectory" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --image_aug False \
  --save_interval "${SAVE_INTERVAL}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --test.type=test-only \
  # --is_resume "${IS_RESUME}" \
  # --resume_step "${RESUME_STEP}" \
  # --resume_epoch "${RESUME_EPOCH}" \
  # --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  # To resume, append the following (and ensure names match the checkpoint):
  # --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  # --is_resume "${IS_RESUME}" \
  # --resume_step "${RESUME_STEP}" \
  # --resume_epoch "${RESUME_EPOCH}"

# load的路径不对 （问题1）
# 所以loss特别高（开始训练的时候）


# test设置
# python vla-scripts/train.py --test.type=test-only --test.test_save_dir="./my_test_results"

