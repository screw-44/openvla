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

# HACK: LeRobot's factory.py does: getattr(importlib.import_module('prismatic.conf.vla'), 'VLAPolicy')
# We register an alias so when prismatic.conf.vla is imported, VLAPolicy is accessible via __getattr__ or direct reference.
# This works because Python's import system caches modules in sys.modules.
import sys

sys.modules["prismatic.conf.vla"].VLAPolicy = VLAPolicy
