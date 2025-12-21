## VLA Policy 集成与 LeRobot-Eval 测评指南

本项目已完成与 LeRobot 的无缝集成，支持直接用 `lerobot-eval` 命令对 VLA Policy 进行评测。以下为完整流程与关键说明。

---

### 1. 检查点转换（如已完成可跳过）

将训练得到的 checkpoint 转换为 LeRobot 兼容格式（包含 config.json 和 model.safetensors）：

```bash
python scripts/convert_checkpoint_to_lerobot.py \
    --input_checkpoint <你的训练checkpoint路径> \
    --output_dir <目标输出目录>
# 例如：
# python scripts/convert_checkpoint_to_lerobot.py --input_checkpoint runs/xxx/checkpoints/latest.pt --output_dir eval_models/vla_latest
```

转换后，`<目标输出目录>` 下应包含：
- config.json  （需包含 type: "vla" 等字段）
- model.safetensors

---

### 2. 运行 LeRobot-Eval 进行测评

核心命令如下：

```bash
lerobot-eval \
    --policy.path=<目标输出目录> \
    --env.type=libero \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --policy.device=cuda \
    --output_dir=./eval_results
```

**参数说明：**
- `--policy.path`：指向包含 config.json 和 model.safetensors 的目录
- `--env.type`：评测环境类型（如 libero、pusht 等）
- `--eval.n_episodes`：评测集 episode 数
- `--eval.batch_size`：评测 batch size
- `--policy.device`：推理设备
- `--output_dir`：评测结果输出目录

---

### 3. 结果查看

评测完成后，结果会保存在 `--output_dir` 指定的目录下，例如：

```
eval_results/2025-12-21/10-30-45_libero_vla/
├── config.yaml         # 完整评测配置
├── metrics.json        # 评测指标（如 success_rate、action_mse 等）
├── rollouts/           # （可选）评测轨迹
└── videos/             # （可选）评测视频
```

metrics.json 示例：
```json
{
  "success_rate": 0.85,
  "average_episode_length": 150,
  "action_mse": 0.0234,
  "action_mae": 0.0156,
  "timestamp": "2025-12-21T10:30:45"
}
```

---

### 4. 常见问题与排查

- **找不到 policy type 'vla'**：
  - 检查 config.json 是否包含 `"type": "vla"`
  - 检查 `pip install -e .` 是否已执行，确保 VLA Policy 已注册
- **模型权重加载失败**：
  - 检查 model.safetensors 是否与 config.json 同目录
- **参数不生效**：
  - 只能在 config.json 或命令行参数中指定，确保字段拼写正确

---

### 5. 参考命令速查

```bash
# 检查点转换
python scripts/convert_checkpoint_to_lerobot.py --input_checkpoint runs/xxx/checkpoints/latest.pt --output_dir eval_models/vla_latest

# 运行评测
lerobot-eval --policy.path=eval_models/vla_latest --env.type=libero --eval.n_episodes=10 --policy.device=cuda

# 查看结果
cat eval_results/2025-12-21/10-30-45_libero_vla/metrics.json
```

---

如需更详细的集成原理、注册机制、FAQ 等，请参考历史文档或联系维护者。我的vla实现嗨嗨嗨
