# Launch Scripts 说明文档

本目录包含所有训练、验证和实验的启动脚本。

## 📁 目录结构

```
launch/
├── base.sh                    # 基础配置脚本（所有脚本的公共配置）
├── basic/                     # 基础功能脚本
│   ├── run.sh                # 纯训练脚本
│   ├── train_with_validate.sh # 训练+验证脚本
│   └── validate.sh           # 独立验证脚本
└── experiments/              # 实验脚本
    └── 0_affordance和表征方式_20251117.sh
```

## 🚀 使用方法

### 1. 基础训练

```bash
# 方法1: 使用默认配置
cd launch/basic
./run.sh

# 方法2: 自定义配置
RUN_ID_NOTE="我的实验" SAVE_INTERVAL=500 ./run.sh
```

### 2. 训练+验证

```bash
cd launch/basic
./train_with_validate.sh

# 自定义验证频率
VALIDATE_INTERVAL=20 NUM_VALIDATION_BATCHES=50 ./train_with_validate.sh
```

### 3. 独立验证

```bash
cd launch/basic

# 方法1: 使用默认checkpoint
./validate.sh

# 方法2: 指定checkpoint路径
./validate.sh /path/to/checkpoint.pt

# 方法3: 使用环境变量
PRETRAINED_CKPT="/path/to/checkpoint.pt" ./validate.sh
```

### 4. 运行实验

```bash
cd launch/experiments
./0_affordance和表征方式_20251117.sh
```

## ⚙️ 配置说明

### base.sh 提供的公共配置

- **自动检测GPU**: 自动检测可用GPU数量
- **环境变量**: OMP_NUM_THREADS, SCRIPT_DIR, PROJECT_ROOT
- **默认路径**: RUN_ROOT_DIR, SAVE_INTERVAL
- **默认参数**: VLA模型类型、数据集配置等

### 可通过环境变量覆盖的配置

所有脚本都支持通过环境变量覆盖默认配置：

```bash
# 示例：自定义所有主要参数
VLA_TYPE="custom-model" \
DATASET_TYPE="custom-dataset" \
RUN_ID_NOTE="实验123" \
SAVE_INTERVAL=100 \
./run.sh
```

#### 通用参数
- `VLA_TYPE`: VLA模型类型（默认: siglip-224px+custom-trajectory）
- `DATASET_TYPE`: 数据集类型（默认: libero）
- `DATASET_REPO`: 数据集仓库（默认: HuggingFaceVLA/libero）
- `TRAJECTORY_COMPRESSION`: 轨迹压缩方式（默认: bining）
- `PROJECT`: WandB项目名称
- `RUN_ID_NOTE`: 运行标识说明
- `SAVE_INTERVAL`: 保存间隔步数（默认: 200）
- `RUN_ROOT_DIR`: 运行结果根目录（默认: runs）

#### 训练+验证专用参数
- `VALIDATE_INTERVAL`: 验证间隔（默认: 10）
- `NUM_VALIDATION_BATCHES`: 每次验证批次数（默认: 30）
- `VALIDATE_SAVE_DIR`: 验证结果保存目录（默认: runs/validation）

#### 验证专用参数
- `PRETRAINED_CKPT`: checkpoint路径
- `VALIDATE_DATA_LENGTH`: 验证数据长度（默认: 10）

## 📝 创建新实验脚本

1. 在 `experiments/` 目录下创建新脚本
2. 使用命名规范: `序号_实验名称_日期.sh`
3. 脚本模板:

```bash
#!/usr/bin/env bash
# 加载基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base.sh"
init_base_config

# 实验特定配置
EXPERIMENT_NAME="你的实验名称"
RUN_ID_NOTE="实验X-${EXPERIMENT_NAME}"

# 自定义配置...

# 启动训练
cd "${PROJECT_ROOT}"
torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} scripts/train.py \
  --mode.type train \
  --vla.type "${VLA_TYPE}" \
  ...
```

## 🎯 最佳实践

1. **使用base.sh**: 所有脚本都应该source base.sh来获取公共配置
2. **环境变量优先**: 通过环境变量传递配置，而不是修改脚本
3. **记录实验**: 实验脚本中包含详细的说明和配置信息
4. **命名规范**: 
   - 基础脚本: 描述性名称（run.sh, validate.sh）
   - 实验脚本: `序号_实验描述_日期.sh`
5. **可重复性**: 确保脚本可以在不同机器上运行

## 💡 中文文件名支持

**是的！Bash完全支持中文文件名！**

- ✅ 文件名可以用中文
- ✅ 注释可以用中文
- ⚠️ 变量名建议用英文（兼容性更好）

示例:
```bash
# ✅ 可以这样
./0_affordance和表征方式_20251117.sh

# ✅ 也可以这样
实验名称="Affordance测试"  # 变量值用中文
echo "${实验名称}"
```

## 🔧 故障排查

### 脚本无法执行
```bash
chmod +x launch/basic/*.sh
chmod +x launch/experiments/*.sh
```

### 找不到base.sh
确保从正确的目录运行，或使用绝对路径

### GPU未检测到
检查nvidia-smi是否正常工作

## 📚 更多信息

- 原始脚本位于项目根目录: `run.sh`, `train_with_validate.sh`, `validate.sh`
- 训练脚本: `scripts/train.py`
- 配置文档: `prismatic/conf/`
