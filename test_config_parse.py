#!/usr/bin/env python
"""
测试配置解析是否正确
"""
import sys

sys.argv = ["test", "--vla.type", "qwen3-vl-2b", "--dataset.type", "libero"]

from dataclasses import dataclass, field
import draccus
from prismatic.conf import VLAConfig, VLARegistry, DatasetConfig, ModeConfig


@dataclass
class TestRunConfig:
    mode: ModeConfig = field(default_factory=lambda: ModeConfig())
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.VLA.vla_id)
    )
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())


cfg = draccus.parse(TestRunConfig)
print(f"VLA ID: {cfg.vla.vla_id}")
print(f"VLA base_vlm: {cfg.vla.base_vlm}")
print(f"Dataset type: {cfg.dataset.type}")
