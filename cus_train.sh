#!/usr/bin/env bash
set -euo pipefail

# 在不能联网的机器上，设置为离线模式
export WANDB_MODE=offline
# 仅本地查找（禁用一切联网）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
  --vla.type "siglip-224px+custom-trajectory" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --image_aug False \
  --save_interval "${SAVE_INTERVAL}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --is_resume "${IS_RESUME}" \
  --resume_step "${RESUME_STEP}" \
  --resume_epoch "${RESUME_EPOCH}" \
  --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  # To resume, append the following (and ensure names match the checkpoint):
  # --pretrained_checkpoint "${PRETRAINED_CKPT}" \
  # --is_resume "${IS_RESUME}" \
  # --resume_step "${RESUME_STEP}" \
  # --resume_epoch "${RESUME_EPOCH}"

# load的路径不对 （问题1）
# 所以loss特别高（开始训练的时候）