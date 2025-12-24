#!/usr/bin/env bash
# ============================================================================
# 测试: Qwen3-VL 模型集成 - 调试模式
# 日期: 2025-12-24
# 目标: 测试 Qwen3-VL 模型加载、训练、图像处理等功能
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# Qwen3-VL 测试配置
# ============================================================================
EXPERIMENT_NAME="test_qwen3_vl_integration"
RUN_ID_NOTE="${EXPERIMENT_NAME}"

# 模型配置 - 使用 Qwen3-VL 2B
VLA_TYPE="qwen3-vl-2b"

# 数据集配置
DATASET_TYPE="libero"
DATASET_REPO="HuggingFaceVLA/libero"
TRAJECTORY_COMPRESSION="action_chunk"  # 显式设置轨迹压缩方法

# 训练配置
SAVE_INTERVAL=500

# 验证配置
VALIDATE_INTERVAL=500
NUM_VALIDATION_BATCHES=10  # 减少验证批次以加快测试

# 训练周期配置（测试用，很少的步数）
EPOCHS=1
MAX_STEPS=100  # 只训练 100 步用于测试

# 项目配置
PROJECT="qwen3-vl-test"

# ============================================================================
# 自动检测 GPU 数量
# ============================================================================
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
echo "检测到 ${NUM_GPUS} 个 GPU"

# ============================================================================
# 启动训练
# ============================================================================
cd "${PROJECT_ROOT}"

echo ""
echo "============================================"
echo "开始测试: ${VLA_TYPE}"
echo "============================================"
echo ""

CURRENT_RUN_ID="${RUN_ID_NOTE}"

echo "最终配置:"
echo "  - VLA Type: ${VLA_TYPE}"
echo "  - Dataset: ${DATASET_TYPE}"
echo "  - Task IDs: [0]"
echo "  - Epochs: ${EPOCHS}"
echo "  - Max Steps: ${MAX_STEPS}"
echo "  - Run ID: ${CURRENT_RUN_ID}"
echo ""

# 启动训练 - 构建命令
TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} scripts/train.py \
  --mode.type train \
  --mode.is_resume false \
  --vla.type \"${VLA_TYPE}\" \
  --dataset.type \"${DATASET_TYPE}\" \
  --dataset.task_ids \"[0]\" \
  --vla.trajectory_compression \"${TRAJECTORY_COMPRESSION}\" \
  --vla.per_device_batch_size 2 \
  --run_root_dir \"${RUN_ROOT_DIR}\" \
  --run_id_note \"${CURRENT_RUN_ID}\" \
  --save_interval \"${SAVE_INTERVAL}\" \
  --epochs ${EPOCHS} \
  --max_steps ${MAX_STEPS} \
  --project \"${PROJECT}\""

# 执行训练
echo "执行命令:"
echo "${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

TRAIN_EXIT_CODE=$?

echo ""
echo "============================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功完成: ${VLA_TYPE}"
else
    echo "❌ 测试失败 (exit code: $TRAIN_EXIT_CODE): ${VLA_TYPE}"
fi

exit $TRAIN_EXIT_CODE
