"""
datasets.py

数据集配置 (Dataset Configuration) - 类似于 VLAConfig 和 ModeConfig 的设计模式

定义了不同的数据集配置及其特定参数：
- LiberoDataset: Libero 数据集
- CustomTrajectoryDataset: 自定义轨迹数据集

使用方式：
    在 RunConfig 中作为嵌套字段:
    dataset: DatasetConfig = field(default_factory=...)
    
    CLI 使用:
    --dataset.type libero
    --dataset.repo_id "HuggingFaceVLA/libero"
    --dataset.task_ids [0,1,2]
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Union

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    """数据集配置基类"""
    dataset_id: str                                             # 数据集唯一标识符
    
    # === Dataset Source ===
    repo_id: Union[str, Path] = "HuggingFaceVLA/libero"        # HF Dataset repo ID 或本地路径
    
    # === Task Selection ===
    task_ids: Optional[List[int]] = None                        # Task IDs to include (None = all tasks)
    
    # === Trajectory Compression ===
    trajectory_compression: str = "bining"                      # Trajectory compression method
    
    def get_task_ids(self) -> Optional[List[int]]:
        """
        获取任务 ID 列表。
        - 返回 None: 加载所有任务的 episodes
        - 返回 List[int]: 只加载指定任务 ID 对应的 episodes
        """
        return self.task_ids


# === 具体的数据集配置 ===

@dataclass
class LiberoDataset(DatasetConfig):
    """Libero 数据集配置"""
    dataset_id: str = "libero"
    repo_id: Union[str, Path] = "HuggingFaceVLA/libero"
    task_ids: Optional[List[int]] = None
    trajectory_compression: str = "bining"


@dataclass
class CustomTrajectoryDataset(DatasetConfig):
    """自定义轨迹数据集配置 - 用于手-物体交互等任务"""
    dataset_id: str = "custom_trajectory"
    repo_id: Union[str, Path] = "local/custom_trajectory"      # 默认本地路径
    task_ids: Optional[List[int]] = None
    trajectory_compression: str = "bining"


@dataclass
class BridgeDataset(DatasetConfig):
    """Bridge 数据集配置"""
    dataset_id: str = "bridge"
    repo_id: Union[str, Path] = "HuggingFaceVLA/bridge"
    task_ids: Optional[List[int]] = None
    trajectory_compression: str = "bining"


# === 数据集注册表 ===
@unique
class DatasetRegistry(Enum):
    """数据集注册表 - 枚举所有可用的数据集配置"""
    LIBERO = LiberoDataset
    CUSTOM_TRAJECTORY = CustomTrajectoryDataset
    BRIDGE = BridgeDataset
    
    @property
    def dataset_id(self) -> str:
        return self.value().dataset_id



# 注册所有数据集配置到 ChoiceRegistry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)

