"""
用lerobot3.0数据集格式，高效率的实现dataset的读取。

使用LeRobotDatasetMetadata先过滤task，然后用LeRobotDataset加载指定的episodes。

核心功能:
1. 支持按task_ids过滤episodes
2. 支持限制每个task加载的episode数量
3. 为每个样本添加future_actions（从当前到episode结束的所有actions）
4. 可配置的处理频率(process_hz)和batch变换
"""
import torch
import numpy as np

from time import time
from datasets import Dataset
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from prismatic.models.backbones.vision.base_vision import ImageTransform
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from prismatic.vla.trajectory_compression import BaseTrajectoryCompression, BiningTrajectoryCompression
from prismatic.vla.tokenizer import VlaTokenizer, BaseTrajectoryConverter
from prismatic.overwatch import initialize_overwatch



# 不同的Dataset有不同的key映射，uniform_key
DATASET_ITEM_MAP_KEYS ={
    'HuggingFaceVLA/libero': {
        'cam1': 'observation.images.image', # 还有 observation.images.image2 (两个camera)
        'cam2': 'observation.images.image2',
        'language': 'task',
    },
}

class MyLeRobotDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            repo_id: str, 
            image_transform: ImageTransform,
            tokenizer: VlaTokenizer,
            trajectory_compression: BaseTrajectoryCompression,
            real_root:Path=Path("/inspire/hdd/project/robot-decision/public/datasets/"), 
            task_ids: list[int] = None,
            train_val_split: Tuple[float, float] = (0.9, 0.1)
        ):
        self.repo_id = repo_id
        self.tokenizer = tokenizer # 都在_get_item__中处理
        self.trajectory_compression = trajectory_compression

        self.root = real_root / repo_id
        self.metadata = LeRobotDatasetMetadata(repo_id, root=self.root)

        # Initialize overwatch logger
        self.overwatch = initialize_overwatch(__name__)

        # 过滤出 task-centric的 episodes；task_ids 为 None 或 [-1] 时加载全部 episodes
        if task_ids is None or task_ids == [-1]:
            self.episodes = list(self.metadata.episodes["episode_index"])
            self.overwatch.info(f"DATASET: Loading ALL episodes ({len(self.episodes)} total)")
        else:
            self.episodes = self.get_episode_indices_for_tasks(task_ids)
            self.overwatch.info(f"DATASET: Loading episodes for task_ids={task_ids} ({len(self.episodes)} episodes)")
            if len(self.episodes) == 0: raise ValueError("No episodes found for the given task_ids; check dataset or filters")
            
        # HACK: 这个属性可以外部修改，决定是拿到训练集还是验证集
        self._get_train_data = True 
        self.train_val_split = train_val_split
        self.train_episode = self.episodes[:int(len(self.episodes)*self.train_val_split[0])] 
        self.val_episode = self.episodes[int(len(self.episodes)*self.train_val_split[0]):]
        self.overwatch.info(f"DATASET: Training episode: {len(self.train_episode)}") #, Validation episode: {len(self.val_episode)}")

        delta_timestamps = {"affordance":[]} if self.is_affordance else None
        self.train_dataset = LeRobotDataset(
            #"HuggingFaceVLA_cus/libero_cut_zcd_20_15_lastseg_indicator",
            repo_id,
            root=self.root,
            episodes=None, # self.train_episode,
            image_transforms=image_transform,
            delta_timestamps=delta_timestamps  # 获取从当前帧到 episode 结尾的完整 action 序列
        )
        # self.val_dataset = LeRobotDataset(
        #     repo_id,
        #     root=self.root,
        #     episodes=self.val_episode,
        #     image_transforms=image_transform,
        #     delta_timestamps=delta_timestamps  # 获取从当前帧到 episode 结尾的完整 action 序列
        # )
        
        self.overwatch.info(f"training dataset length:{len(self.train_dataset)}") #, validate dataset length:{len(self.val_dataset)}")

    @property
    def is_affordance(self):
        return "aff" in self.trajectory_compression.exp_type

    @property
    def get_train_data(self):
        """获取当前使用的数据集类型（训练/验证）"""
        return self._get_train_data
    
    @get_train_data.setter
    def get_train_data(self, value: bool):
        """设置使用训练集还是验证集"""
        self._get_train_data = value
    
    @property
    def dataset(self):
        """动态返回训练集或验证集"""
        return self.train_dataset # if self._get_train_data else self.val_dataset
    
    def get_episode_indices_for_tasks(self, task_ids: list[int]) -> list[int]:
        tasks = self.metadata.tasks
        # 不同的repo的实现是不同的，注意这里 TODO: 未来分成不同的类
        if self.repo_id.endswith("libero"):
            # 对于libero的而言，根据meta中的文本string来过滤出task_id
            # tasks 的 index 通常是 task_name（string），所以需要先获取对应的 task_name
            task_mask = tasks["task_index"].isin(task_ids)
            selected_task_str = tasks[task_mask].index.tolist()  # 获取选中的 task_name list
            
            selected_episode_metadata = self.metadata.episodes.filter(lambda x: x['tasks'][0] in selected_task_str)
            result = list(selected_episode_metadata["episode_index"])
            
            return result
        elif self.repo_id.endswith("pusht_image"):
            pass
        elif self.repo_id.endswith("2025-challenge-demos"):
            pass
        else: 
            raise NotImplementedError(f"Unknown repo_id format: {self.repo_id}")
    
    def get_trajectory_for_item(self, item):
        # affordance 已经从数据集中作为 tensor 字段直接获取
        qurey_key = "affordance" if self.is_affordance else "action"
        original_trajectory = item[qurey_key].numpy()
        compressed_trajectory = self.trajectory_compression(original_trajectory)
        return torch.Tensor(compressed_trajectory)

    def __len__(self): 
        return len(self.train_dataset)
        # 返回当前数据集（训练或验证）的正确长度，而不是使用 LeRobotDataset 的长度（它总是返回全部数据）
        # if self._get_train_data:
        #     return len(self.train_dataset)
        # else:
        #     return len(self.val_dataset)
    
    def __getitem__(self, index):
        # 根据是哪一个具体的数据集，拿到对应的数据
        item = self.dataset.__getitem__(index)

        # 这里扩展到了两图输入的libero的格式（目前先focus在libero上）
        uni_key_item = dict(
            cam1=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam1']],
            cam2=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['cam2']],
            language=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['language']],
            trajectory=self.get_trajectory_for_item(item),
            dataset_names=self.repo_id
        )

        return self.tokenizer.tokenize_batch(uni_key_item)
    