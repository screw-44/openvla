"""
vla.py

注意这个里面的参数除了需要满足训练的需求，还有hugginface的代码来进行推理

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
"""

from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

import torch


from lerobot.configs.policies import PreTrainedConfig


# 尝试只注册夫类来实现
@PreTrainedConfig.register_subclass("vla")
@dataclass
class VLAConfig(PreTrainedConfig):
    # fmt: off
    vla_id: str = "base"                            # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path] = "distilgpt2"  # Base VLM as ID/Path to Run Directory
    freeze_vision_backbone: bool = True             # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool = False               # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool = False           # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Optimization Parameters (epochs and max_steps moved to RunConfig in train.py)
    global_batch_size: int = -1                     # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int = 32                 # Per-Device Batch Size (per-process / individual GPU) of accumulation steps is auto-computed

    learning_rate: float = 5e-4                     # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float = 0.001                     # Weight Decay for AdamW Optimizer
    max_grad_norm: float = 2.0                      # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str = "linear-warmup+cosine-decay"  # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float = 0.01                      # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str = "fsdp-full-shard"         # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # Trajectory Configuration
    trajectory_compression: str = "bining"          # Trajectory compression method (e.g., 'bining', 'action_chunk', 'uniform_bspline')
    trajectory_converter_type: str = 'value_textualize'  # Converter type for action discretization
    trajectory_n_bins: int = 256                    # Number of bins for discretization
    trajectory_n_dims: int = 7                      # Action dimensions (e.g., 7DOF for Libero)

    ### ===== 推理部分的代码（hf lerobot） ======
    type: str = "vla"
    action_dim: int = 7
    action_horizon: int = 1
    observation_horizon: int = 1
    vision_backbone: str = "siglip-vit-so400m-14-384"
    llm_backbone: str = "distilgpt2"
    model_config: dict = field(default_factory=dict) # Model Configuration (serialized from checkpoint)

    # fmt: on

    def __post_init__(self) -> None:
        """
        Post-initialization hook to compute global_batch_size if it's set to -1 (auto-compute mode).
        For VLA training, global_batch_size must equal per_device_batch_size * number_of_gpus
        to ensure grad_accumulation_steps == 1.
        """
        if self.global_batch_size == -1:
            # Auto-compute global_batch_size based on per_device_batch_size and available GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.global_batch_size = self.per_device_batch_size * num_gpus

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> None:
        return None

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        return None


@dataclass
class Base_4090Config(VLAConfig):
    vla_id: str = "base_4090"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    per_device_batch_size: int = 1


# === Debug Models for Fast Iteration ===
@dataclass
class DistilGPT2Config(VLAConfig):
    vla_id: str = "distilgpt2"
    base_vlm: Union[str, Path] = "distilgpt2"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = True

    warmup_ratio: float = 0.01  # 0.01% warmup
    learning_rate: float = 5e-4  # Slightly higher LR for faster convergence
    weight_decay: float = 0.001
    # Very small batch for quick testing
    per_device_batch_size: int =  64
    global_batch_size: int = -1  # Auto-compute


# === Qwen3-VL Models ===
@dataclass
class Qwen3VL_2BConfig(VLAConfig):
    vla_id: str = "qwen3-vl-2b"
    base_vlm: Union[str, Path] = "qwen3-vl-2b"

    freeze_vision_backbone: bool = False  # Qwen3-VL handles vision internally
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    per_device_batch_size: int = 32
    global_batch_size: int = -1


@dataclass
class Qwen3VL_7BConfig(Qwen3VL_2BConfig):
    vla_id: str = "qwen3-vl-7b"
    base_vlm: Union[str, Path] = "qwen3-vl-7b"
    per_device_batch_size: int = 16


@dataclass
class Qwen3VL_4BConfig(Qwen3VL_2BConfig):
    vla_id: str = "qwen3-vl-4b"
    base_vlm: Union[str, Path] = "qwen3-vl-4b"
    per_device_batch_size: int = 16


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    VLA = VLAConfig
    # === Custom Trajectory Training ===
    Base_4090 = Base_4090Config

    # === Debug Models ===
    DISTILGPT2 = DistilGPT2Config

    # === Qwen3-VL Models ===
    QWEN3_VL_2B = Qwen3VL_2BConfig
    QWEN3_VL_7B = Qwen3VL_7BConfig
    QWEN3_VL_4B = Qwen3VL_4BConfig

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
