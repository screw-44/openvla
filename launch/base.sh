#!/usr/bin/env bash
# ============================================================================
# Base Configuration Script
# 所有训练/验证脚本的公共配置
# ============================================================================

set -euo pipefail

# ============================================================================
# GPU 配置
# ============================================================================
auto_detect_gpus() {
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "Error: No GPUs detected!"
        exit 1
    fi
    echo "Detected ${NUM_GPUS} GPU(s), will use all of them."
    export NUM_GPUS
}

# ============================================================================
# 环境变量配置
# ============================================================================
setup_environment() {
    # OpenMP 线程设置
    export OMP_NUM_THREADS=1
    
    # Resolve script directory
    export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
}

# ============================================================================
# 路径配置
# ============================================================================
setup_paths() {
    # 默认路径配置
    export RUN_ROOT_DIR="${RUN_ROOT_DIR:-runs}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
}

# ============================================================================
# 通用配置
# ============================================================================
# VLA 模型类型
export DEFAULT_VLA_TYPE="siglip-224px+custom-trajectory"

# 数据集配置
export DEFAULT_DATASET_TYPE="libero"
export DEFAULT_DATASET_REPO="HuggingFaceVLA/libero"
export DEFAULT_TRAJECTORY_COMPRESSION="bining"

# 训练参数（epochs 和 max_steps 现在是 RunConfig 的一部分）
export DEFAULT_EPOCHS=1000
export DEFAULT_MAX_STEPS=""  # 留空表示使用 epochs，设置值则覆盖 epochs

# 项目名称
export DEFAULT_PROJECT="vla-hand-object"

# ============================================================================
# 初始化函数 - 在其他脚本中调用
# ============================================================================
init_base_config() {
    auto_detect_gpus
    setup_environment
    setup_paths
    
    echo "============================================"
    echo "Base Configuration Initialized"
    echo "============================================"
    echo "GPUs: ${NUM_GPUS}"
    echo "Project Root: ${PROJECT_ROOT}"
    echo "Run Root Dir: ${RUN_ROOT_DIR}"
    echo "Save Interval: ${SAVE_INTERVAL}"
    echo "============================================"
}
