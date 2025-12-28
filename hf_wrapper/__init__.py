"""
LeRobot-compatible VLA Policy wrapper.

Provides VLAPolicy and VLAConfig for use with lerobot-eval framework.
The configuration class is automatically registered via @PreTrainedConfig.register_subclass("vla")
and can be discovered dynamically by the LeRobot factory.
"""

from .configuration_vla import VLAConfig
from .modeling_vla import VLAPolicy
from .processor_vla import make_vla_pre_post_processors

__all__ = [
    "VLAConfig",
    "VLAPolicy",
    "make_vla_pre_post_processors",
]
