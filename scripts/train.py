"""
train.py - VLA Training Script with Hydra + OmegaConf

Hydra handles all configuration management automatically.

Usage Examples:
    python scripts/train.py                                    # Use default config
    python scripts/train.py vla=qwen2.5-0.5b                  # Switch VLA variant
    python scripts/train.py vla=qwen2.5-0.5b vla.optimization.learning_rate=1e-4
    python scripts/train.py --help                            # Show all available options
    python scripts/train.py --cfg job                         # Print resolved config
"""

import re

from pathlib import Path
from typing import Any

import hydra
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig

from core.models import load
from core.util.overwatch import initialize_overwatch
from core.training import VLAMetrics, get_train_strategy
from core.util import set_global_seed
from core.util.vla_utils import get_vla_dataset
from utils.training_utils import find_latest_checkpoint, warmup_trainig

from hydra.core.hydra_config import HydraConfig

# Initialize early (before Hydra takes over)
local_rank = warmup_trainig()
overwatch = initialize_overwatch(__name__)


@hydra.main(version_base=None, config_path="/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/config", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function - Hydra handles all config loading and composition."""
    overwatch.info("OpenVLA Training :: Starting")
    
    # === Print Configuration ===
    overwatch.info("=" * 80)
    overwatch.info("FULL CONFIGURATION (Hydra + OmegaConf):")
    overwatch.info(OmegaConf.to_yaml(cfg))
    overwatch.info("=" * 80)
    
    # === Extract VLA, Mode, Dataset configs ===
    vla_cfg = cfg.vla
    mode_cfg = cfg.mode
    dataset_cfg = cfg.dataset
    is_validate = cfg.is_validate
    
    overwatch.info(f"✅ Config loaded: vla_id={vla_cfg.vla_id}, lr={vla_cfg.optimization.learning_rate}")

    # === Generate Run ID ===
    vla_id = vla_cfg.vla_id
    run_id = cfg.run_id or f"{vla_id}+b{vla_cfg.optimization.per_device_batch_size}+x{cfg.seed}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    
    # === Setup Directories and Randomness ===
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    run_dir = Path(HydraConfig.get().run.dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # === Save Configuration ===
    if overwatch.is_rank_zero():
        # Save full Hydra config as YAML (for training reference)
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        overwatch.info(f"Config saved to {run_dir / 'config.yaml'}")
        
        # Save minimal config.json for HuggingFace/LeRobot inference
        import json
        hf_config = {
            "vla_id": vla_cfg.vla_id,
            "model_id": vla_cfg.get("model_id", vla_cfg.vla_id),
            "base_vlm": vla_cfg.get("base_vlm", vla_cfg.vla_id),
            "type": "vla",
            "trajectory_compression": vla_cfg.trajectory.compression_method,
            "trajectory_converter_type": vla_cfg.trajectory.converter_type,
            "trajectory_n_bins": vla_cfg.trajectory.n_bins,
            "trajectory_n_dims": vla_cfg.trajectory.n_dims,
            "action_dim": vla_cfg.get("action_dim", 7),
            "action_horizon": vla_cfg.get("action_horizon", 1),
            "observation_horizon": vla_cfg.get("observation_horizon", 1),
            "model_config": {},
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(hf_config, f, indent=2)
        overwatch.info(f"HuggingFace config saved to {run_dir / 'config.json'}")

    # === Load Checkpoint (if resuming) ===
    checkpoint_to_load = None
    if mode_cfg.is_resume:
        checkpoint_to_load = (
            find_latest_checkpoint(run_dir)
            if mode_cfg.pretrained_checkpoint is None
            else Path(mode_cfg.pretrained_checkpoint)
        )
        if checkpoint_to_load is None:
            raise ValueError("No checkpoint found, but mode.is_resume=True")

    # === Load VLM ===
    overwatch.info(
        f"Loading VLM (checkpoint={checkpoint_to_load}, validation={is_validate})"
    )
    vla = load(
        vla_cfg=vla_cfg,
        checkpoint_path=checkpoint_to_load,
        load_for_training=not is_validate,
    )

    # === Determine Training Stage ===
    freeze_vision = vla_cfg.freeze_vision_backbone
    freeze_llm = vla_cfg.freeze_llm_backbone
    unfreeze_last = vla_cfg.unfreeze_last_llm_layer
    
    if not freeze_vision and not freeze_llm:
        stage = "vla-full-train"
    elif freeze_vision and not freeze_llm:
        stage = "vla-train"
    elif not freeze_vision and freeze_llm:
        assert unfreeze_last, "Must unfreeze at least last LLM layer!"
        stage = "vla-sandwich-train"
    elif freeze_vision and freeze_llm:
        assert unfreeze_last, "Must unfreeze at least last LLM layer!"
        stage = "vla-last-layer-train"
    else:
        raise ValueError(f"Invalid freeze config: vision={freeze_vision}, llm={freeze_llm}")
    
    overwatch.info(f"Training stage: {stage}")
    vla.freeze_backbones(stage)

    # === Print Model Size ===
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(f"Model: {num_params/1e6:.1f}M params, {num_trainable/1e6:.1f}M trainable")

    # === Create VLA Dataset ===
    overwatch.info("Creating VLA Dataset...")
    vla_dataset, trajectory_converter, collator = get_vla_dataset(
        data_repo_id=dataset_cfg.repo_id,
        data_task_ids=dataset_cfg.get_task_ids() if hasattr(dataset_cfg, 'get_task_ids') else None,
        trajectory_compression_method=vla_cfg.trajectory.compression_method,
        trajectory_converter_type=vla_cfg.trajectory.converter_type,
        trajectory_n_bins=vla_cfg.trajectory.n_bins,
        trajectory_n_dims=vla_cfg.trajectory.n_dims,
        base_tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        image_transform=vla.vision_backbone.get_image_transform(),
    )

    # === Extract Resume State ===
    resume_step, resume_epoch = 0, 0
    if mode_cfg.is_resume and checkpoint_to_load is not None:
        filename = checkpoint_to_load.name
        step_match = re.search(r"step-(\d+)", filename)
        epoch_match = re.search(r"epoch-(\d+)", filename)
        resume_step = int(step_match.group(1)) if step_match else 0
        resume_epoch = int(epoch_match.group(1)) if epoch_match else 0
        overwatch.info(f"Resuming: step={resume_step}, epoch={resume_epoch}")

    # === Create Train Strategy ===
    overwatch.info(f"Initializing train strategy: {vla_cfg.optimization.train_strategy}")
    train_strategy = get_train_strategy(
        train_strategy=vla_cfg.optimization.train_strategy,
        vla=vla,
        device_id=local_rank,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=vla_cfg.optimization.global_batch_size,
        per_device_batch_size=vla_cfg.optimization.per_device_batch_size,
        learning_rate=vla_cfg.optimization.learning_rate,
        weight_decay=vla_cfg.optimization.weight_decay,
        max_grad_norm=vla_cfg.optimization.max_grad_norm,
        lr_scheduler_type=vla_cfg.optimization.lr_scheduler_type,
        warmup_ratio=vla_cfg.optimization.warmup_ratio,
        enable_gradient_checkpointing=vla_cfg.optimization.enable_gradient_checkpointing,
        enable_mixed_precision_training=vla_cfg.optimization.enable_mixed_precision_training,
        reduce_in_full_precision=vla_cfg.optimization.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(
        run_dir=run_dir,
        n_train_examples=len(vla_dataset),
        is_resume=mode_cfg.is_resume,
    )

    # === Setup Metrics ===
    metrics = VLAMetrics(
        run_id,
        OmegaConf.to_container(cfg, resolve=True),
        group="vla-train",
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        project=cfg.project,
    )

    # === Run Training or Validation ===
    overwatch.info("Starting Training Loop...")
    train_strategy.train_vla(
        vla_dataset,
        collator,
        metrics,
        run_dir=run_dir,
        save_interval=cfg.save_interval,
    )
    overwatch.info("Training Complete")

    metrics.finalize()
    dist.barrier()
    dist.destroy_process_group()


def cleanup_empty_runs(runs_base_dir: Path) -> None:
    """
    遍历 outputs/{日期}/{时间} 目录结构
    如果某个时间目录下的所有模型都不包含 .safetensors 文件，则删除该时间目录
    """
    runs_base_dir = Path(runs_base_dir)
    import shutil
    # 遍历所有日期目录
    for date_dir in runs_base_dir.iterdir():
        if not date_dir.is_dir():
            continue

        # 记录需要删除的空时间目录
        empty_time_dirs = []
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            # 检查这个时间目录下是否存在任何 .safetensors 文件
            has_safetensors = any(time_dir.rglob("*.safetensors"))
            if not has_safetensors:
                empty_time_dirs.append(time_dir)

        # 删除所有空时间目录
        for time_dir in empty_time_dirs:
            try:
                shutil.rmtree(time_dir)
                overwatch.info(f"Deleted empty run directory: {time_dir}")
            except FileNotFoundError:
                pass

        # 如果该日期目录下已无任何时间目录，则删除该日期目录
        if not any(date_dir.iterdir()):
            try:
                shutil.rmtree(date_dir)
                overwatch.info(f"Deleted empty date directory: {date_dir}")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    if overwatch.is_rank_zero():
        cleanup_empty_runs("/inspire/ssd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla/outputs/")
    train()
