"""
configuration_vla.py

Minimal VLAConfig for HuggingFace/LeRobot inference.
This is a lightweight config that only contains parameters needed for model loading and inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("vla")
@dataclass
class VLAConfig(PreTrainedConfig):
    """
    Minimal VLA Configuration for HuggingFace inference.
    
    This config only contains essential parameters needed for:
    1. Model loading (base_vlm, vla_id)
    2. Trajectory decoding (trajectory_compression, n_bins, n_dims, etc.)
    3. LeRobot compatibility (action_dim, action_horizon, etc.)
    
    Training-specific parameters (optimization, freezing, etc.) are NOT included here.
    They are managed by Hydra configs during training.
    """
    # === Model Identification ===
    vla_id: str = "base"
    model_id: str = "base"  # Used by VLA() for model initialization
    base_vlm: Union[str, Path] = "distilgpt2"
    type: str = "vla"
    
    # === Trajectory Configuration (needed for decoding) ===
    trajectory_compression: str = "action_chunk"
    trajectory_converter_type: str = "value_textualize"
    trajectory_n_bins: int = 256
    trajectory_n_dims: int = 7
    
    # === LeRobot Inference Fields ===
    action_dim: int = 7
    action_horizon: int = 1
    observation_horizon: int = 1
    
    # === Model Config (serialized from checkpoint) ===
    model_config: dict = field(default_factory=dict)
    
    # === Optional Properties for LeRobot Compatibility ===
    @property
    def observation_delta_indices(self):
        return None
    
    @property
    def action_delta_indices(self):
        return None
    
    @property
    def reward_delta_indices(self):
        return None
    
    def get_optimizer_preset(self):
        return None
    
    def get_scheduler_preset(self):
        return None
    
    def validate_features(self):
        return None


__all__ = ["VLAConfig"]