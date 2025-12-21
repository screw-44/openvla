"""
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

import torch
from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters (epochs and max_steps moved to RunConfig in train.py)
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # Trajectory Converter Configuration
    trajectory_converter_type: str = 'value_textualize'  # Converter type for action discretization
    trajectory_n_bins: int = 256                    # Number of bins for discretization
    trajectory_n_dims: int = 7                      # Action dimensions (e.g., 7DOF for Libero)

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


# === [1 GPU] Lightweight Custom Trajectory Training ===
@dataclass
class Base(VLAConfig):
    vla_id: str = "base"
    base_vlm: Union[str, Path] = "TRI-ML/prismatic-vlms/siglip-224px+7b"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False
    shuffle_buffer_size: int = 256_000  # Smaller buffer for lightweight training

    # H100 Settings
    # Note: global_batch_size = -1 means auto-compute: per_device_batch_size * number_of_gpus
    global_batch_size: int = -1  # Auto-computed in __post_init__
    per_device_batch_size: int = 32

    learning_rate: float = 5e-4  # Slightly higher LR for faster convergence
    weight_decay: float = 0.001  # Add some regularization
    max_grad_norm: float = 2.0
    lr_scheduler_type: str = "linear-warmup+cosine-decay"
    warmup_ratio: float = 0.01  # 0.0001     # 0.01% warmup

    train_strategy: str = "fsdp-full-shard"

    use_flash_attention_2: bool = True
    enable_mixed_precision_training: bool = True


@dataclass
class Base_4090(Base):
    vla_id: str = "base_4090"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    per_device_batch_size: int = 1


# === Debug Models for Fast Iteration ===
@dataclass
class DistilGPT2(Base):
    vla_id: str = "distilgpt2"
    base_vlm: Union[str, Path] = "distilgpt2"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = True

    learning_rate: float = 5e-4  # Slightly higher LR for faster convergence
    warmup_ratio: float = 0.01  # 0.01% warmup
    weight_decay: float = 0.001
    # Very small batch for quick testing
    per_device_batch_size: int = 64
    global_batch_size: int = -1  # Auto-compute


# === Qwen3-VL Models ===
@dataclass
class Qwen3VL_2B(Base):
    vla_id: str = "qwen3-vl-2b"
    base_vlm: Union[str, Path] = "qwen3-vl-2b"

    freeze_vision_backbone: bool = False  # Qwen3-VL handles vision internally
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    per_device_batch_size: int = 32
    global_batch_size: int = -1


@dataclass
class Qwen3VL_7B(Qwen3VL_2B):
    vla_id: str = "qwen3-vl-7b"
    base_vlm: Union[str, Path] = "qwen3-vl-7b"
    per_device_batch_size: int = 16


@dataclass
class Qwen3VL_4B(Qwen3VL_2B):
    vla_id: str = "qwen3-vl-4b"
    base_vlm: Union[str, Path] = "qwen3-vl-4b"
    per_device_batch_size: int = 16


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # === Custom Trajectory Training ===
    Base = Base
    Base_4090 = Base_4090

    # === Debug Models ===
    DISTILGPT2 = DistilGPT2

    # === Qwen3-VL Models ===
    QWEN3_VL_2B = Qwen3VL_2B
    QWEN3_VL_7B = Qwen3VL_7B
    QWEN3_VL_4B = Qwen3VL_4B

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
