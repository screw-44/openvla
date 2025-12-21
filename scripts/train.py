"""
train.py
目前暂时不修改这个名字，暂时认为全量微调是更好的训练方式，lora的代码未来自己进行实现。
"""

import re
import os
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import draccus
import torch.distributed as dist

# 不再从 prismatic.conf.run 导入 RunConfig
from prismatic.conf import (
    VLAConfig,
    VLARegistry,
    ModeConfig,
    ModeRegistry,
    DatasetConfig,
    DatasetRegistry,
)
from prismatic.models import load
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from utils.training_utils import find_latest_checkpoint, warmup_trainig

local_rank = warmup_trainig()
overwatch = initialize_overwatch(__name__)


# === RunConfig 定义（类似原始 OpenVLA 的设计）===
@dataclass
class RunConfig:
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
        default_factory=DatasetConfig.get_choice_class(
            DatasetRegistry.LIBERO.dataset_id
        )
    )
    # === Directory Paths ===
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    # === Run Arguments ===
    run_id: Optional[str] = None  # Run ID for logging
    run_id_note: Optional[str] = None  # Extra note for logging
    save_interval: int = 2500  # Interval for saving checkpoints (steps)
    seed: int = 7  # Random seed
    # === Training Duration Parameters ===
    epochs: int = 100  # Epochs to Run (in case max_steps is not specified)
    max_steps: Optional[int] = (
        None  # [Optional] Max Gradient Steps to Run (overrides epochs)
    )
    # === Trackio Project Configuration ===
    project: str = "vla-training"  # Trackio project name


@draccus.wrap()
def train(cfg: RunConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+b{cfg.vla.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories and Set Randomness
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "train_config.yml", "w"))

    # 加载模型
    checkpoint_to_load = None
    if cfg.mode.is_resume:
        checkpoint_to_load = (
            find_latest_checkpoint(run_dir)
            if cfg.mode.pretrained_checkpoint == None
            else cfg.mode.pretrained_checkpoint
        )
        if checkpoint_to_load is None:
            raise ValueError("No checkpoint found, But cfg mode is resume==True")
    overwatch.info(
        f"Loading VLM: path:{checkpoint_to_load}, load_for_trainig: {not cfg.mode.is_validate}"
    )
    vla = load(
        vla_cfg=cfg.vla,
        checkpoint_path=checkpoint_to_load,
        load_for_training=not cfg.mode.is_validate,
    )

    # 冻结参数 Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert (
            cfg.vla.unfreeze_last_llm_layer
        ), "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert (
            cfg.vla.unfreeze_last_llm_layer
        ), "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            f"Weight freezing configuration not supported. VLA config has the following parameters: freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}, freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}, unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )
    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(
        f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`"
    )
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

    # Extract resume_step and resume_epoch from checkpoint path if resuming
    if cfg.mode.is_resume and checkpoint_to_load is not None:
        filename = checkpoint_to_load.name
        step_match, epoch_match = re.search(r"step-(\d+)", filename), re.search(
            r"epoch-(\d+)", filename
        )
        resume_step, resume_epoch = int(step_match.group(1)) if step_match else 0, (
            int(epoch_match.group(1)) if epoch_match else 0
        )
        overwatch.info(
            f"Resuming from checkpoint: step={resume_step}, epoch={resume_epoch}"
        )
    else:
        resume_step, resume_epoch = 0, 0

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.vla.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.vla.train_strategy,
        vla=vla,
        device_id=local_rank,  # 开头指定了
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
    train_strategy.run_setup(
        run_dir=run_dir, n_train_examples=len(vla_dataset), is_resume=cfg.mode.is_resume
    )

    metrics = VLAMetrics(
        cfg.run_id,
        draccus.encode(cfg),
        group="vla-train",
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        project=cfg.project,
    )

    # Run VLA Training Loop or Testing
    if cfg.mode.is_validate:
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
        )
        overwatch.info("Done with Training =>> Finalizing Metrics")

    metrics.finalize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
