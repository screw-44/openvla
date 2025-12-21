#!/usr/bin/env bash
# ============================================================================
# 实验：OpenVLA Baseline 训练
# 日期: 2025-12-17
# 目标: 使用标准的 OpenVLA 训练方式，在所有 LIBERO tasks 上训练并测试效果
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 实验特定配置
# ============================================================================
EXPERIMENT_NAME="baseline_openvla_12_17"
RUN_ID_NOTE="${EXPERIMENT_NAME}"

# 模型配置
VLA_TYPE="base"

# 数据集配置
DATASET_TYPE="libero"
DATASET_REPO="HuggingFaceVLA/libero"
DATASET_TASK_IDS="[-1]"  # -1 表示使用所有 tasks

# 训练配置
SAVE_INTERVAL=1000

# 验证配置
VALIDATE_INTERVAL=500  # 每 500 步验证一次
NUM_VALIDATION_BATCHES=500  # 每次验证使用的批次数

# 训练周期配置（epochs 和 max_steps 现在是 RunConfig 的一部分）
EPOCHS="${EPOCHS:-1}"  # 实验用较少的 epochs  :-代表默认数值
MAX_STEPS="${MAX_STEPS:-}"  # 留空使用 epochs

# 项目配置
PROJECT="vla-affordance-experiment"

# ============================================================================
# 实验说明
# ============================================================================
echo "============================================"
echo "实验: OpenVLA Baseline 训练"
echo "============================================"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "日期: 2025-12-17"
echo ""
echo "实验配置:"
echo "  - VLA Type: ${VLA_TYPE}"
echo "  - Dataset: ${DATASET_TYPE}"
echo "  - Task IDs: 所有 tasks ([-1])"
echo "  - Save Interval: ${SAVE_INTERVAL}"
echo "  - Validate Interval: ${VALIDATE_INTERVAL}"
echo "  - Validation Batches: ${NUM_VALIDATION_BATCHES}"
echo ""
echo "实验目标:"
echo "  1. 使用标准的 OpenVLA 训练方式"
echo "  2. 在所有 LIBERO tasks 上训练模型"
echo "  3. 测试 action_chunk 方法的效果"
echo "  4. 为后续实验建立 baseline"
echo "============================================"

# ============================================================================
# 启动训练 - OpenVLA Baseline
# ============================================================================
cd "${PROJECT_ROOT}"

# 使用 action_chunk 作为 trajectory compression 方法（OpenVLA 标准配置）
COMPRESSION_METHODS=(
    "action_chunk"           # OpenVLA 标准的 action chunking 方法
)

echo "============================================"
echo "使用 trajectory compression 方法: action_chunk"
echo "训练数据: LIBERO 所有 tasks"
echo "============================================"
echo ""

# 循环测试每种压缩方法
for COMPRESSION_METHOD in "${COMPRESSION_METHODS[@]}"; do
    echo "============================================"
    echo "开始实验: ${COMPRESSION_METHOD}"
    echo "============================================"
    
    CURRENT_RUN_ID="${RUN_ID_NOTE}-${COMPRESSION_METHOD}"
    
    echo "配置:"
    echo "  - Compression Method: ${COMPRESSION_METHOD}"
    echo "  - Run ID: ${CURRENT_RUN_ID}"
    echo "  - VLA Type: ${VLA_TYPE}"
    echo "  - Dataset: ${DATASET_TYPE}"
    echo "  - Task IDs: 所有 tasks ([-1])"
    echo "  - Epochs: ${EPOCHS}"
    if [ -n "${MAX_STEPS}" ]; then
        echo "  - Max Steps: ${MAX_STEPS} (overrides epochs)"
    fi
    echo ""
    
    # 启动训练 - 构建命令
    TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} scripts/train.py \
      --mode.type train_validate \
      --mode.is_resume false \
      --mode.validate_interval ${VALIDATE_INTERVAL} \
      --mode.num_validation_batches ${NUM_VALIDATION_BATCHES} \
      --vla.type \"${VLA_TYPE}\" \
      --dataset.type \"${DATASET_TYPE}\" \
      --dataset.repo_id \"${DATASET_REPO}\" \
      --dataset.task_ids [-1] \
      --dataset.trajectory_compression \"${COMPRESSION_METHOD}\" \
      --run_root_dir \"${RUN_ROOT_DIR}\" \
      --run_id_note \"${CURRENT_RUN_ID}\" \
      --save_interval \"${SAVE_INTERVAL}\" \
      --epochs ${EPOCHS} \
      --project \"${PROJECT}\""
    
    # 如果设置了 MAX_STEPS，添加该参数
    if [ -n "${MAX_STEPS}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --max_steps ${MAX_STEPS}"
    fi
    
    # 执行训练
    eval ${TRAIN_CMD}
    
    echo ""
    echo "============================================"
    echo "完成实验: ${COMPRESSION_METHOD}"
    echo "============================================"
    echo ""
    
    # 可选: 在实验之间添加短暂延迟
    # sleep 5
done

echo "============================================"
echo "OpenVLA Baseline 训练完成！"
echo "============================================"
echo ""
echo "训练配置:"
echo "  ✓ Compression Method: action_chunk"
echo "  ✓ Dataset: LIBERO (所有 tasks)"
echo "  ✓ VLA Type: ${VLA_TYPE}"
echo ""
echo "结果保存在: ${RUN_ROOT_DIR}"
echo "============================================"
