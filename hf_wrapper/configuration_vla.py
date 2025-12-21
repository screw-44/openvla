"""
VLA Policy Configuration for LeRobot.

This module defines VLAConfig which is automatically registered with LeRobot's
policy factory through the @PreTrainedConfig.register_subclass("vla") decorator.
"""

from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("vla")
@dataclass
class VLAConfig(PreTrainedConfig):
    """
    Configuration class for VLA Policy.
    
    This configuration enables VLA policies to be automatically discovered and loaded
    by LeRobot without modifying the factory code. The decorator registration handles
    dynamic discovery.
    
    Attributes:
        type: Policy type identifier ("vla")
        action_dim: Dimension of action space (default: 7 for ALOHA-like tasks)
        action_horizon: Number of action steps to predict
        observation_horizon: Number of observation steps to use
        vision_backbone: Vision encoder backbone (e.g., "timm-vit-so400m-14-384")
        llm_backbone: Language model backbone (e.g., "distilgpt2")
        device: Device for inference ("cpu", "cuda", etc.)
        dtype: Data type for model (e.g., "bfloat16", "float32")
    """
    
    type: str = "vla"
    
    # ============ Model Architecture ============
    action_dim: int = 7
    action_horizon: int = 4
    observation_horizon: int = 2
    
    vision_backbone: str = "siglip-vit-so400m-14-384"
    llm_backbone: str = "distilgpt2"
    
    # ============ Training Parameters ============
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    
    # ============ Device & Precision ============
    device: Optional[str] = None  # Will auto-select in __post_init__
    dtype: str = "bfloat16"
    use_amp: bool = False
    
    # ============ Model Configuration (serialized from checkpoint) ============
    model_config: dict = field(default_factory=dict)
    
    # ============ Required by PreTrainedConfig ============
    
    @property
    def observation_delta_indices(self) -> list | None:
        """LeRobot framework requirement: indices for observation delta computation."""
        return None
    
    @property
    def action_delta_indices(self) -> list | None:
        """LeRobot framework requirement: indices for action delta computation."""
        return None
    
    @property
    def reward_delta_indices(self) -> list | None:
        """LeRobot framework requirement: indices for reward delta computation."""
        return None
    
    def get_optimizer_preset(self):
        """LeRobot framework requirement: return default optimizer config."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    
    def get_scheduler_preset(self):
        """LeRobot framework requirement: return default scheduler config."""
        return None
    
    def validate_features(self):
        """LeRobot framework requirement: validate input/output feature definitions."""
        # VLA can work with minimal features, no strict validation needed
        pass
