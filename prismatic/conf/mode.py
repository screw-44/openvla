"""
mode.py

运行模式配置 (Mode Configuration) - 类似于 VLAConfig 的设计模式

定义了不同的运行模式及其特定参数：
- TrainMode: 纯训练模式
- TrainValidateMode: 训练+验证模式  
- TestMode: 测试模式

使用方式：
    在 RunConfig 中作为嵌套字段:
    mode: ModeConfig = field(default_factory=...)
    
    CLI 使用:
    --mode.type train  # 或 train_validate, test
    --mode.test_save_dir "runs/test"
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class ModeConfig(ChoiceRegistry):
    """运行模式配置基类"""
    mode_id: str                                    # 运行模式唯一标识符

    # === Resume Run Parameters ===
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run

    # 测试/验证相关参数（所有模式都可能用到）
    validate_save_dir: Optional[Path] = Path("runs/validation_results")  # 测试/验证结果保存目录
    validate_data_length: int = -1                      # 测试/验证batch数量 (-1 = 全部)
    validate_checkpoint_path: Optional[Path] = None          # 指定checkpoint路径

    # === 便捷属性 ===
    @property
    def is_test(self) -> bool: return self.mode_id == "test"
    
    @property
    def is_validate(self) -> bool: return "validate" in self.mode_id
    
    @property
    def is_train(self) -> bool: return "train" in self.mode_id


# === 具体的运行模式配置 ===

@dataclass
class TrainMode(ModeConfig):
    """纯训练模式 - 只训练不验证"""
    mode_id: str = "train"


@dataclass
class TrainValidateMode(ModeConfig):
    """训练+验证模式 - 训练过程中定期验证"""
    mode_id: str = "train_validate"
    
    # 验证特定参数
    validate_interval: int = 1000                   # 验证间隔（步数）
    num_validation_batches: int = 30               # 每次验证的batch数量
    
    def __post_init__(self):
        # 验证模式默认值设置
        if self.validate_data_length == -1:
            self.validate_data_length = self.num_validation_batches


@dataclass
class TestMode(ModeConfig):
    """测试模式 - 仅评估不训练"""
    mode_id: str = "test"


# === 运行模式注册表 ===
@unique
class ModeRegistry(Enum):
    """运行模式注册表 - 枚举所有可用的运行模式"""
    TRAIN = TrainMode
    TRAIN_VALIDATE = TrainValidateMode
    TEST = TestMode
    
    @property
    def mode_id(self) -> str: return self.value().mode_id


# 注册所有运行模式到 ChoiceRegistry
for mode_variant in ModeRegistry:
    ModeConfig.register_subclass(mode_variant.mode_id, mode_variant.value)
