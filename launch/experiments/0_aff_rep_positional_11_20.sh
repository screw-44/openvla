#!/usr/bin/env bash
# ============================================================================
# 实验：Affordance 和表征方式对比实验
# 日期: 2025-11-17
# 目标: 测试不同的 affordance 表征方式对模型性能的影响
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 实验特定配置
# ============================================================================
EXPERIMENT_NAME="aff_rep_251120"
RUN_ID_NOTE="${EXPERIMENT_NAME}"

# 模型配置
VLA_TYPE="base"

# 数据集配置
DATASET_TYPE="libero"
DATASET_REPO="HuggingFaceVLA/libero"
# 注意: TRAJECTORY_COMPRESSION 将在循环中测试多种方法

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

# 启动训练 - 测试不同的 trajectory_compression 方法
# ============================================================================
cd "${PROJECT_ROOT}"

# 定义要测试的所有 trajectory_compression 方法
COMPRESSION_METHODS=(
    # "positional_bining"
    "positional_uniform_bspline"
)

echo "============================================"
echo "将依次测试以下 trajectory compression 方法:"
for method in "${COMPRESSION_METHODS[@]}"; do
    echo "  - ${method}"
done
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
    echo "  - Epochs: ${EPOCHS}"
    if [ -n "${MAX_STEPS}" ]; then
        echo "  - Max Steps: ${MAX_STEPS} (overrides epochs)"
    fi
    echo ""
    
    # 启动训练 - 构建命令
    TRAIN_CMD="torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} vla-scripts/train.py \
      --mode.type train_validate \
      --mode.is_resume false \
      --mode.validate_interval ${VALIDATE_INTERVAL} \
      --mode.num_validation_batches ${NUM_VALIDATION_BATCHES} \
      --vla.type \"${VLA_TYPE}\" \
      --dataset.type \"${DATASET_TYPE}\" \
      --dataset.repo_id \"${DATASET_REPO}\" \
      --dataset.trajectory_compression \"${COMPRESSION_METHOD}\" \
      --run_root_dir \"${RUN_ROOT_DIR}\" \
      --run_id_note \"${CURRENT_RUN_ID}\" \
      --save_interval \"${SAVE_INTERVAL}\" \
      --epochs ${EPOCHS} \
      --project \"${PROJECT}\" \
      "

      #\
      # --vla.per_device_batch_size 1
    
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
echo "所有 trajectory compression 方法测试完成！"
echo "============================================"
echo ""
echo "测试的方法:"
for method in "${COMPRESSION_METHODS[@]}"; do
    echo "  ✓ ${method}"
done
echo ""
echo "结果保存在: ${RUN_ROOT_DIR}"
echo "============================================"
