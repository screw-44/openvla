"""
用lerobot3.0数据集格式，高效率的实现dataset的读取。

使用LeRobotDatasetMetadata先过滤task，然后用LeRobotDataset加载指定的episodes。

核心功能:
1. 支持按task_ids过滤episodes
2. 支持限制每个task加载的episode数量
3. 为每个样本添加future_actions（从当前到episode结束的所有actions）
4. 可配置的处理频率(process_hz)和batch变换
"""
import logging
import torch
import numpy as np

from time import time
from datasets import Dataset
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from prismatic.vla.trajectory_compression import BaseTrajectoryCompression, BiningTrajectoryCompression
from prismatic.vla.tokenizer import VlaTokenizer, BaseTrajectoryConverter



# 不同的Dataset有不同的key映射，uniform_key
DATASET_ITEM_MAP_KEYS ={
    'HuggingFaceVLA/libero': {
        'rgb': 'observation.images.image', # 还有observation.images.image2 (两个camera)
        'language': 'task',
    },
}

class MyLeRobotDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            repo_id: str, 
            tokenizer: VlaTokenizer,
            trajectory_compression: BaseTrajectoryCompression,
            real_root:Path=Path("/inspire/hdd/project/robot-decision/public/datasets/"), 
            task_ids: list[int] = [0],
            train_val_split: Tuple[float, float] = (0.9, 0.1)
        ):
        # TODO： downsample hz: process_hz: int = 10, 
        self.repo_id = repo_id
        self.tokenizer = tokenizer # 都在_get_item__中处理
        self.trajectory_compression = trajectory_compression

        self.root = real_root / repo_id
        self.metadata = LeRobotDatasetMetadata(repo_id, root=self.root)
        # logger
        self.logger = logging.getLogger(__name__)

        # 过滤出 task-centric的 episodes
        self.episodes = None if task_ids == [-1] else self.get_episode_indices_for_tasks(task_ids)
        
        # 分成不同的训练数据集和验证数据集，测试直接在仿真里面测试
        self.train_val_split = train_val_split
        self._get_train_data = True # HACK: 这个属性可以外部修改，决定是拿到训练集还是验证集
        self.train_episode = self.episodes[:int(len(self.episodes)*self.train_val_split[0])] 
        self.val_episode = self.episodes[int(len(self.episodes)*self.train_val_split[0]):]
        self.logger.info(f"DATASET: Training episode: {len(self.train_episode)}, Validation episode: {len(self.val_episode)}")
        
        self.train_dataset = LeRobotDataset(
            repo_id,
            root=self.root,
            episodes=self.train_episode
        )
        self.val_dataset = LeRobotDataset(
            repo_id,
            root=self.root,
            episodes=self.val_episode
        )


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
        return self.train_dataset if self._get_train_data else self.val_dataset

    def get_episode_indices_for_tasks(self, task_ids: list[int]) -> list[int]:
        tasks = self.metadata.tasks
        # 不同的repo的实现是不同的，注意这里 TODO：
        if self.repo_id.endswith("libero"):
            # 对于libero的而言，根据meta中的文本string来过滤出task_id
            selected_task_str = tasks[tasks["task_index"].isin(task_ids)].index.tolist()
            selected_episode_metadata = self.metadata.episodes.filter(lambda x: x['tasks'][0] in selected_task_str)
            return list(selected_episode_metadata["episode_index"])
        elif self.repo_id.endswith("pusht_image"):
            pass
        elif self.repo_id.endswith("2025-challenge-demos"):
            pass
        else: 
            raise NotImplementedError(f"Unknown repo_id format: {self.repo_id}")
    
    def get_trajectory_for_item(self, item):
        episode_id, frame_id = item['episode_index'].item(), item['frame_index'].item() # tensor转换成int
    
        # 注意是dataset的meta，而不是空的metadata
        episode_from_id, episode_to_id = self.dataset.meta.episodes['dataset_from_index'][episode_id], self.dataset.meta.episodes['dataset_to_index'][episode_id]
        original_trajectory = np.array(self.dataset.hf_dataset['action'][episode_from_id+frame_id:episode_to_id])
        compressed_trajectory = self.trajectory_compression(original_trajectory)
        return torch.Tensor(compressed_trajectory)

    def __len__(self):      
        return len(self.dataset)
    
    def __getitem__(self, index):
        # 根据是哪一个具体的数据集，拿到对应的数据
        item = self.dataset.__getitem__(index)

        # TODO: 目前是单个图像输入，应该扩展到单+多个图像的输入
        uni_key_item = dict(
            rgb=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['rgb']],
            language=item[DATASET_ITEM_MAP_KEYS[self.repo_id]['language']],
            trajectory=self.get_trajectory_for_item(item),
            dataset_names=self.repo_id
        )
        
        return self.tokenizer.tokenize_batch(uni_key_item)
    


# 内存cache的实现，优先级很低，先整理代码，未来再合理实现
# trajectory 的内存cache {} episode_id -> trajectory {original, compressed{frame -> trajectory}: ...}
# 数据量很小，整个lerobot的数据是一个1g。同时，一次运行后会缓存到数据集的目录下，下次直接读取缓存
# self.trajectory_file_path = self.root / "trajectory.arrow"
# if self.trajectory_file_path.exists():
#     self.logger.info(f"Loading cached trajectory dataset from {self.trajectory_file_path}")
#     self.trajectory_dataset = Dataset.from_file(self.trajectory_file_path)
# else:
#     # 初始化空的trajectory dataset，后续动态添加
#     self.logger.info(f"Initializing empty trajectory dataset")
#     self.trajectory_dataset = Dataset.from_dict({
#         'episode_index': [],
#         'frame_id': [],
#         'original_trajectory': [],  # 可以存储为list或者序列化的数据
#         'compressed_trajectory': []
#     })

# def save_trajectory_dataset(self):
#     self.trajectory_dataset.save_to_disk(self.trajectory_file_path)

# Arrow不支持直接存储2维的数据，所以转换成list成一维. 且必须传入原子类型
# trajectory_data = {
#     'episode_index': episode_id,
#     'frame_id': frame_id,
#     'original_trajectory': [x.tolist() for x in original_trajectory],
#     'compressed_trajectory': [x.tolist() for x in compressed_trajectory]
# }
# self.trajectory_dataset = self.trajectory_dataset.add_item(trajectory_data)