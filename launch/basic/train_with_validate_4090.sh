#!/usr/bin/env bash
# ============================================================================
# 训练+验证脚本
# 使用方法: ./train_with_validate.sh
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 训练+验证特定配置
# ============================================================================
VLA_TYPE="base_4090"
DATASET_TYPE="${DATASET_TYPE:-${DEFAULT_DATASET_TYPE}}"
DATASET_REPO="${DATASET_REPO:-${DEFAULT_DATASET_REPO}}"
TRAJECTORY_COMPRESSION="${TRAJECTORY_COMPRESSION:-${DEFAULT_TRAJECTORY_COMPRESSION}}"
PROJECT="${PROJECT:-${DEFAULT_PROJECT}}"
RUN_ID_NOTE="${RUN_ID_NOTE:-train-validate}"

# 训练周期配置
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
MAX_STEPS=6

# 验证配置
VALIDATE_INTERVAL=10          # 每N步验证一次
NUM_VALIDATION_BATCHES=10 # 每次验证使用的批次数
VALIDATE_SAVE_DIR="${VALIDATE_SAVE_DIR:-runs/validation}" # 验证结果保存路径

# 从pretrianed load吗
IS_RESUME=true
PRETRAINED_CHECKPOINT="/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/runs/base+b32+x7--aff_representation_251117-action_chunk/checkpoints/latest-checkpoint.pt"

# 保存频率
SAVE_INTERVAL=5 # 每N个epoch保存一次模型

# ============================================================================
# 启动训练+验证
# ============================================================================
echo "Starting training with validation:"
echo "  VLA Type: ${VLA_TYPE}"
echo "  Dataset: ${DATASET_TYPE}"
echo "  Validate Interval: ${VALIDATE_INTERVAL}"
echo "  Validation Batches: ${NUM_VALIDATION_BATCHES}"
echo "  Run ID Note: ${RUN_ID_NOTE}"
echo "  Epochs: ${EPOCHS}"
if [ -n "${MAX_STEPS}" ]; then
    echo "  Max Steps: ${MAX_STEPS} (overrides epochs)"
fi
echo "============================================"

cd "${PROJECT_ROOT}"

# 构建命令参数
TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} vla-scripts/train.py \
  --mode.type train_validate \
  --mode.is_resume ${IS_RESUME} \
  --mode.pretrained_checkpoint ${PRETRAINED_CHECKPOINT} \
  --mode.validate_interval ${VALIDATE_INTERVAL} \
  --mode.num_validation_batches ${NUM_VALIDATION_BATCHES} \
  --mode.validate_save_dir \"${VALIDATE_SAVE_DIR}\" \
  --vla.type \"${VLA_TYPE}\" \
  --dataset.type \"${DATASET_TYPE}\" \
  --dataset.repo_id \"${DATASET_REPO}\" \
  --dataset.trajectory_compression \"${TRAJECTORY_COMPRESSION}\" \
  --run_root_dir \"${RUN_ROOT_DIR}\" \
  --run_id_note \"${RUN_ID_NOTE}\" \
  --save_interval \"${SAVE_INTERVAL}\" \
  --epochs ${EPOCHS} \
  --project \"${PROJECT}\""

# 如果设置了 MAX_STEPS，添加该参数
if [ -n "${MAX_STEPS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max_steps ${MAX_STEPS}"
fi

# 执行训练
eval ${TRAIN_CMD}
