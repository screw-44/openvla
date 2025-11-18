#!/usr/bin/env bash
# ============================================================================
# 验证/测试脚本
# 使用方法: ./validate.sh [checkpoint_path]
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 验证特定配置
# ============================================================================
VLA_TYPE="${VLA_TYPE:-${DEFAULT_VLA_TYPE}}"
DATASET_TYPE="${DATASET_TYPE:-${DEFAULT_DATASET_TYPE}}"
DATASET_REPO="${DATASET_REPO:-${DEFAULT_DATASET_REPO}}"
TRAJECTORY_COMPRESSION="${TRAJECTORY_COMPRESSION:-${DEFAULT_TRAJECTORY_COMPRESSION}}"
PROJECT="${PROJECT:-vla-validation}"

# Checkpoint 路径（可以通过命令行参数或环境变量指定）
if [ -n "${1:-}" ]; then
    PRETRAINED_CKPT="$1"
elif [ -z "${PRETRAINED_CKPT:-}" ]; then
    # 默认 checkpoint 路径
    PRETRAINED_CKPT="${PROJECT_ROOT}/runs/siglip-224px+custom-trajectory+n0+b1+x7/checkpoints/step-033999-epoch-00-loss=0.3088.pt"
fi

# 验证配置
VALIDATE_DATA_LENGTH="${VALIDATE_DATA_LENGTH:-10}"
VALIDATE_SAVE_DIR="${VALIDATE_SAVE_DIR:-runs/test_results}"

# ============================================================================
# 启动验证
# ============================================================================
echo "Starting validation/testing:"
echo "  Checkpoint: ${PRETRAINED_CKPT}"
echo "  VLA Type: ${VLA_TYPE}"
echo "  Dataset: ${DATASET_TYPE}"
echo "  Data Length: ${VALIDATE_DATA_LENGTH}"
echo "  Save Dir: ${VALIDATE_SAVE_DIR}"
echo "============================================"

cd "${PROJECT_ROOT}"

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} vla-scripts/train.py \
  --mode.type test \
  --mode.validate_checkpoint_path "${PRETRAINED_CKPT}" \
  --mode.validate_data_length ${VALIDATE_DATA_LENGTH} \
  --mode.validate_save_dir "${VALIDATE_SAVE_DIR}" \
  --vla.type "${VLA_TYPE}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --dataset.type "${DATASET_TYPE}" \
  --dataset.repo_id "${DATASET_REPO}" \
  --dataset.trajectory_compression "${TRAJECTORY_COMPRESSION}" \
  --project "${PROJECT}"
