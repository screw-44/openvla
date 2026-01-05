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
    """空的config, 具体参数在modeling中loading yaml"""
    type: str = field(default="vla", init=False)  # 让 draccus 识别配置类型
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