"""
train.py
目前暂时不修改这个名字，暂时认为全量微调是更好的训练方式，lora的代码我们保留但是不进行主线开发。

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). 

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, List

import draccus
import torch
import torch.distributed as dist
import yaml

# 不再从 prismatic.conf.run 导入 RunConfig
from prismatic.conf import VLAConfig, VLARegistry, ModeConfig, ModeRegistry, DatasetConfig, DatasetRegistry
from prismatic.models import load
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 分布式训练奇怪的设定 ---
# 强制根据 Local Rank 设置当前进程可见的 GPU
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# [TODO: Fix for NCCL Error] Explicitly initialize process group with device_id
# This prevents "using GPU X as device used by this process is currently unknown"
if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}")
    )

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Utility Function: Extract Step and Epoch from Checkpoint Path ===
def extract_step_epoch_from_checkpoint(checkpoint_path: Optional[Path]) -> Tuple[int, int]:
    """
    从 checkpoint 文件路径中提取 step 和 epoch 信息
    
    预期文件名格式示例:
        - step-010000-epoch-01-loss=0.2341.pt
        - step-005000-epoch-00.pt
        
    Args:
        checkpoint_path: checkpoint 文件路径
        
    Returns:
        (resume_step, resume_epoch) 元组，如果无法解析则返回 (0, 0)
    """
    if checkpoint_path is None:
        return 0, 0
    
    filename = checkpoint_path.name
    
    # 尝试匹配 step-XXXXXX-epoch-XX 格式
    step_match = re.search(r'step-(\d+)', filename)
    epoch_match = re.search(r'epoch-(\d+)', filename)
    
    resume_step = int(step_match.group(1)) if step_match else 0
    resume_epoch = int(epoch_match.group(1)) if epoch_match else 0
    
    return resume_step, resume_epoch


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    在指定目录下查找最新的 checkpoint 文件
    
    Args:
        run_dir: 运行目录（包含 checkpoints 子目录）
        
    Returns:
        最新的 checkpoint 文件路径，如果没有找到则返回 None
    """
    checkpoint_dir = run_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        overwatch.info(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # 查找所有 .pt 文件
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        overwatch.info(f"No checkpoint files found in: {checkpoint_dir}")
        return None
    
    # 提取每个文件的 step 数字，找到最大的
    checkpoints_with_steps = []
    for ckpt_path in checkpoint_files:
        step, _ = extract_step_epoch_from_checkpoint(ckpt_path)
        checkpoints_with_steps.append((step, ckpt_path))
    
    # 按 step 排序，取最大的
    checkpoints_with_steps.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoints_with_steps[0][1]
    
    overwatch.info(f"Found latest checkpoint: {latest_checkpoint.name} (step={checkpoints_with_steps[0][0]})")
    return latest_checkpoint


# === RunConfig 定义（类似原始 OpenVLA 的设计）===
@dataclass
class RunConfig:
    """
    顶层运行配置 
    
    将运行模式 (ModeConfig)、VLA 配置和数据集配置作为嵌套字段
    利用 ChoiceRegistry 的优势，保持 CLI 参数灵活性
    
    CLI 使用示例:
        --mode.type train                      # 选择训练模式
        --mode.type test                       # 选择测试模式  
        --vla.type "siglip-224px+..."         # VLA 模型选择
        --dataset.type libero       # 数据集选择
        --dataset.repo_id "huggingface id"      # 数据集路径
    """
    # === 运行模式配置（嵌套 ChoiceRegistry）===
    mode: ModeConfig = field(
        default_factory=ModeConfig.get_choice_class(ModeRegistry.TRAIN.mode_id)
    )
    # === VLA Model Configuration（嵌套 ChoiceRegistry）===
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.Base.vla_id)
    )
    # === Dataset Configuration（嵌套 ChoiceRegistry）===
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LIBERO.dataset_id)
    )

    # === Directory Paths ===
    run_root_dir: Path = Path("runs")                           # Path to directory to store logs & checkpoints
    # === Run Arguments ===
    run_id: Optional[str] = None                                # Run ID for logging
    run_id_note: Optional[str] = None                           # Extra note for logging
    save_interval: int = 2500                                   # Interval for saving checkpoints (steps)
    seed: int = 7                                               # Random seed
    # === Training Duration Parameters ===
    epochs: int = 100                                           # Epochs to Run (in case max_steps is not specified)
    max_steps: Optional[int] = None                             # [Optional] Max Gradient Steps to Run (overrides epochs)
    
    # === Trackio Project Configuration ===
    project: str = "vla-training"                               # Trackio project name
    
    @property
    def test_save_dir(self) -> Optional[Path]:
        """测试/验证保存目录"""
        return self.mode.validate_save_dir
    
    @property
    def test_data_length(self) -> int:
        """测试/验证数据长度"""
        return self.mode.validate_data_length
    
    @property  
    def validate_interval(self) -> int:
        """验证间隔"""
        if hasattr(self.mode, 'validate_interval'):
            return self.mode.validate_interval
        return 1000


@draccus.wrap()
def train(cfg: RunConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = f"{vla_id}+b{cfg.vla.per_device_batch_size}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    
    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Auto-find latest checkpoint if is_resume=True but pretrained_checkpoint is None
    checkpoint_to_load = cfg.mode.pretrained_checkpoint
    if cfg.mode.is_resume and checkpoint_to_load is None:
        overwatch.info("is_resume=True but pretrained_checkpoint not specified, searching for latest checkpoint...")
        checkpoint_to_load = find_latest_checkpoint(run_dir)
        if checkpoint_to_load is None:
            overwatch.info("No checkpoint found, will load base VLM instead")

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if checkpoint_to_load is not None:
        if cfg.mode.is_resume:
            print("[*] Loading VLA from Pretrained Checkpoint:", checkpoint_to_load)
            vla = load(cfg.vla, checkpoint_to_load, load_for_training=True)
        elif cfg.mode.is_test:
            print("[*] Loading VLA from Checkpoint for Validate/Test:", cfg.mode.validate_checkpoint_path)
            vla = load(cfg.vla, load_for_training=False)
    else:
        vla = load(cfg.vla, load_for_training=True)

    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Dataset. ")
    vla_dataset, trajectory_converter, collator = get_vla_dataset_and_collator(
        data_repo_id=cfg.dataset.repo_id,
        data_task_ids=cfg.dataset.get_task_ids(),
        trajectory_compression_method=cfg.dataset.trajectory_compression,
        trajectory_converter_type=cfg.vla.trajectory_converter_type,
        trajectory_n_bins=cfg.vla.trajectory_n_bins,
        trajectory_n_dims=cfg.vla.trajectory_n_dims,
        base_tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        image_transform=vla.vision_backbone.get_image_transform(),
        default_image_resolution=vla.vision_backbone.default_image_resolution,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.vla.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.vla.train_strategy,
        vla=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.vla.global_batch_size,
        per_device_batch_size=cfg.vla.per_device_batch_size,
        learning_rate=cfg.vla.learning_rate,
        weight_decay=cfg.vla.weight_decay,
        max_grad_norm=cfg.vla.max_grad_norm,
        lr_scheduler_type=cfg.vla.lr_scheduler_type,
        warmup_ratio=cfg.vla.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # Extract resume_step and resume_epoch from checkpoint path if resuming
    if cfg.mode.is_resume and checkpoint_to_load is not None:
        filename = checkpoint_to_load.name
        step_match, epoch_match = re.search(r'step-(\d+)', filename), re.search(r'epoch-(\d+)', filename)
        resume_step, resume_epoch = int(step_match.group(1)) if step_match else 0, int(epoch_match.group(1)) if epoch_match else 0
        overwatch.info(f"Resuming from checkpoint: step={resume_step}, epoch={resume_epoch}")
    else:
        resume_step, resume_epoch = 0, 0
    
    metrics = VLAMetrics(
        cfg.run_id,
        draccus.encode(cfg),
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        project=cfg.project,
    )

    # Run VLA Training Loop or Testing
    if cfg.mode.is_test:
        overwatch.info("Starting VLA Valite Loop")
        train_strategy.validate_vla(
            vla_dataset,
            collator,
            metrics,
            mode_config=cfg.mode,
        )
        overwatch.info("Done with Validate")
    else:
        overwatch.info("Starting VLA Training Loop") 
        train_strategy.train_vla(
            vla_dataset,
            collator,
            metrics,
            run_dir=run_dir,
            save_interval=cfg.save_interval,
            mode_config=cfg.mode,
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
        )
        overwatch.info("Done with Training =>> Finalizing Metrics")
    
    metrics.finalize()
    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
