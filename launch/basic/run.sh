#!/usr/bin/env bash
# ============================================================================
# 基础训练脚本（纯训练模式）
# 使用方法: ./run.sh
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 训练特定配置
# ============================================================================
VLA_TYPE="${VLA_TYPE:-${DEFAULT_VLA_TYPE}}"
DATASET_TYPE="${DATASET_TYPE:-${DEFAULT_DATASET_TYPE}}"
DATASET_REPO="${DATASET_REPO:-${DEFAULT_DATASET_REPO}}"
TRAJECTORY_COMPRESSION="${TRAJECTORY_COMPRESSION:-${DEFAULT_TRAJECTORY_COMPRESSION}}"
PROJECT="${PROJECT:-${DEFAULT_PROJECT}}"
RUN_ID_NOTE="${RUN_ID_NOTE:-测试}"

# 训练周期配置
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"

# ============================================================================
# 启动训练
# ============================================================================
echo "Starting training with the following configuration:"
echo "  VLA Type: ${VLA_TYPE}"
echo "  Dataset: ${DATASET_TYPE}"
echo "  Run ID Note: ${RUN_ID_NOTE}"
echo "  Epochs: ${EPOCHS}"
if [ -n "${MAX_STEPS}" ]; then
    echo "  Max Steps: ${MAX_STEPS} (overrides epochs)"
fi
echo "============================================"

cd "${PROJECT_ROOT}"

# 构建命令参数
TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} scripts/train.py \
  --mode.type train \
  --mode.is_resume false \
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
