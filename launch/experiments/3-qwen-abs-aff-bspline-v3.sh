#!/usr/bin/env bash
# ============================================================================
# 测试: Qwen3-VL 模型集成 - 调试模式
# 日期: 2025-12-18
# 目标: 测试 Qwen3-VL 模型加载、多GPU训练、梯度检查点等功能
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 测试特定配置
# ============================================================================
EXPERIMENT_NAME='1-qwen25-abs_aff_uniform_bspline_v3'
RUN_ID_NOTE="${EXPERIMENT_NAME}"

# 模型配置
VLA_TYPE="qwen2.5-0.5b"

# 数据集配置
DATASET_TYPE="libero"
DATASET_REPO="HuggingFaceVLA/libero"
TRAJECTORY_COMPRESSION="bspline_v3"  # 显式设置轨迹压缩方法

# 训练配置
SAVE_INTERVAL=5000

# 训练周期配置
EPOCHS=10

# 项目配置
PROJECT="bspline_v3" # 测试就放在useless这里

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
echo "  - Epochs: ${EPOCHS}"
echo "  - Run ID: ${CURRENT_RUN_ID}"
echo ""

# 启动训练 - 构建命令
TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} scripts/train.py \
    vla=${VLA_TYPE} \
    mode=train \
    mode.is_resume=false \
    dataset=${DATASET_TYPE} \
    vla.trajectory.compression_method=${TRAJECTORY_COMPRESSION} \
    vla.trajectory.converter_type=${TRAJECTORY_COMPRESSION} \
    run_root_dir=${RUN_ROOT_DIR} \
    run_id_note=\"${CURRENT_RUN_ID}\" \
    save_interval=${SAVE_INTERVAL} \
    vla.optimization.per_device_batch_size=16 \
    vla.optimization.global_batch_size=64 \
    epochs=${EPOCHS} \
    project=${PROJECT}"


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
