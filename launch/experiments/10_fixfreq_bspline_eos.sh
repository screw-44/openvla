#!/usr/bin/env bash
# ============================================================================
# 实验：Fix Freq Uniform B-Spline + EOS Token 训练
# 日期: 2025-12-17
# 目标: 使用 fix_freq_uniform_bspline 压缩方法 + EOS token 实现变长轨迹生成
# ============================================================================

# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"

# 初始化配置
init_base_config

# ============================================================================
# 实验特定配置
# ============================================================================
EXPERIMENT_NAME="fixfreq_bspline_eos_12_17"
RUN_ID_NOTE="${EXPERIMENT_NAME}"

# 模型配置
VLA_TYPE="base"

# 数据集配置
DATASET_TYPE="libero"
DATASET_REPO="HuggingFaceVLA/libero"
DATASET_TASK_IDS="[-1]"  # -1 表示使用所有 tasks

# 轨迹压缩方法
COMPRESSION_METHOD="fix_freq_uniform_bspline"

# 训练配置
SAVE_INTERVAL=1000

# 验证配置
VALIDATE_INTERVAL=500  # 每 500 步验证一次
NUM_VALIDATION_BATCHES=500  # 每次验证使用的批次数 500

# 训练周期配置
EPOCHS=3
MAX_STEPS="${MAX_STEPS:-}"  # 留空使用 epochs

# 项目配置
PROJECT="vla-affordance-experiment"

# ============================================================================
# 实验说明
# ============================================================================
echo "============================================"
echo "实验: Fix Freq Uniform B-Spline + EOS Token 训练"
echo "============================================"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "日期: 2025-12-17"
echo ""
echo "实验配置:"
echo "  - VLA Type: ${VLA_TYPE}"
echo "  - Dataset: ${DATASET_TYPE}"
echo "  - Task IDs: 所有 tasks ([-1])"
echo "  - Compression Method: ${COMPRESSION_METHOD}"
echo "  - Save Interval: ${SAVE_INTERVAL}"
echo "  - Validate Interval: ${VALIDATE_INTERVAL}"
echo "  - Validation Batches: ${NUM_VALIDATION_BATCHES}"
echo ""
echo "实验目标:"
echo "  1. 使用 fix_freq_uniform_bspline B样条拟合压缩方法"
echo "  2. 实现变长轨迹生成 (no padding, 使用 EOS token)"
echo "  3. 测试时间优先的 token 顺序 (t0_d0, t0_d1, ..., t1_d0, ...)"
echo "  4. 在所有 LIBERO tasks 上训练并验证效果"
echo ""
echo "技术特点:"
echo "  - 拟合: 使用最小二乘法拟合 B-spline 曲线"
echo "  - 压缩: 提取控制点作为轨迹表示 (target_length + degree + 1 个点)"
echo "  - 截断: 根据 frame_percentage 动态截断控制点"
echo "  - 结束: 添加 EOS token 标记轨迹结束"
echo "  - 优势: 相比 bining，B-spline 能更好保留轨迹平滑性"
echo "  - 对比: 与 fix_freq_bining 对比 (fitting vs sampling)"
echo "============================================"

# ============================================================================
# 启动训练
# ============================================================================
cd "${PROJECT_ROOT}"

echo ""
echo "============================================"
echo "开始训练: ${COMPRESSION_METHOD}"
echo "============================================"
echo ""

CURRENT_RUN_ID="${RUN_ID_NOTE}"

echo "最终配置:"
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
  mode=train \
  mode.is_resume=false \
  mode.validate_interval=${VALIDATE_INTERVAL} \
  mode.num_validation_batches=${NUM_VALIDATION_BATCHES} \
  vla=\"${VLA_TYPE}\" \
  dataset=\"${DATASET_TYPE}\" \
  --dataset.repo_id \"${DATASET_REPO}\" \
  --dataset.task_ids [-1] \
  --dataset.trajectory_compression \"${COMPRESSION_METHOD}\" \
  run_root_dir=\"${RUN_ROOT_DIR}\" \
  run_id_note=\"${CURRENT_RUN_ID}\" \
  save_interval=\"${SAVE_INTERVAL}\" \
  epochs=${EPOCHS} \
  project=\"${PROJECT}\""

# 如果设置了 MAX_STEPS，添加该参数
if [ -n "${MAX_STEPS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} max_steps=${MAX_STEPS}"
fi

# 执行训练
eval ${TRAIN_CMD}

echo ""
echo "============================================"
echo "训练完成: ${COMPRESSION_METHOD}"
echo "============================================"
echo ""
echo "训练结果:"
echo "  ✓ Compression Method: ${COMPRESSION_METHOD}"
echo "  ✓ Dataset: LIBERO (所有 tasks)"
echo "  ✓ VLA Type: ${VLA_TYPE}"
echo "  ✓ 实现特性: 变长轨迹 + EOS token + B-spline 平滑"
echo ""
echo "结果保存在: ${RUN_ROOT_DIR}/${CURRENT_RUN_ID}"
echo ""
echo "后续步骤:"
echo "  1. 查看训练日志和 metrics"
echo "  2. 对比 fix_freq_bining vs fix_freq_uniform_bspline"
echo "  3. 分析 B-spline 拟合对轨迹质量的影响"
echo "  4. 评估模型在 LIBERO 仿真环境中的表现"
echo "  5. 对比不同时间步 (t0, t0-t5) 的准确率"
echo ""
echo "B-Spline 优势分析:"
echo "  - 更少的控制点表示更长的轨迹 (压缩率更高)"
echo "  - 保持轨迹平滑性 (避免采样点的锯齿效应)"
echo "  - 最小二乘法拟合能更好处理噪声数据"
echo "============================================"
