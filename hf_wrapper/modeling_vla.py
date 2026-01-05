"""
VLA Policy implementation for LeRobot.

This module implements VLAPolicy which is automatically discovered by LeRobot
through the naming convention: VLAConfig (in configuration_vla.py) → VLAPolicy.

The factory will automatically:
1. Find VLAConfig in PreTrainedConfig.get_known_choices()
2. Infer the module path: configuration_vla → modeling_vla
3. Infer the class name: VLAConfig → VLAPolicy
4. Dynamically import and instantiate VLAPolicy
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction
from transformers import GenerationMixin

from prismatic.models.load import load
from .configuration_vla import VLAConfig
from prismatic.models.vlms.vla import VLA
from prismatic.util.vla_utils import get_vla_tokenizer
from prismatic.vla.tokenizer import BaseTrajectoryConverter, VlaTokenizer
from prismatic.vla.trajectory_compression import BaseTrajectoryCompression
from prismatic.models.vlms.base_vlm import VLM
from prismatic.vla.dataset import DATASET_ITEM_MAP_KEYS

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class VLAPolicy(PreTrainedPolicy):
    """
    VLA Policy compatible with LeRobot evaluation framework.
    """

    config_class = VLAConfig
    name = "vla"

    def __init__(self, cfg: DictConfig, config: VLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.cfg = cfg
        self.model: VLA = None

        self.trajectory_converter: BaseTrajectoryConverter = None
        self.vla_tokenizer: VlaTokenizer = None
        self.trajectory_compression: BaseTrajectoryCompression = None

        self.dataset_name = "HuggingFaceVLA/libero"
        self.decode_control_point = None
        self.reconstruct_traj = None
        self.current_step = 0
        self.last_step = 0
        self.last_action = (0, 0, 0, 0, 0, 0, -1) # 初始位置的位置

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: Union[str, Path],
        *,
        config: Optional[VLAConfig] = None,
        **kwargs,
    ) -> "VLAPolicy":
        """
        Load VLA policy from checkpoint.

        Args:
            pretrained_name_or_path: Path to checkpoint directory
            config: Optional VLAConfig (loaded from config.json if not provided)
        """
        pretrained_name_or_path = Path(pretrained_name_or_path)
        import yaml
        config_path = pretrained_name_or_path / "config.yaml"
        with open(config_path) as f:
            cfg = DictConfig(yaml.safe_load(f))

        # Create policy instance
        policy = cls(cfg, config, **kwargs)

        # Load model weights
        policy.model = load(
            vla_cfg=cfg.vla,
            checkpoint_path=pretrained_name_or_path / "model.safetensors",
            load_for_training=False,
        )
        policy.model.to("cuda").eval()  # 默认load cuda

        logger.info(f"Loading from the function: load is complete")

        (
            policy.trajectory_converter,
            policy.vla_tokenizer,
            policy.trajectory_compression,
        ) = get_vla_tokenizer(
            trajectory_compression_method=cfg.vla.trajectory.compression_method,
            base_tokenizer=policy.model.llm_backbone.tokenizer,
            prompt_builder_fn=policy.model.llm_backbone.prompt_builder_fn,
            trajectory_converter_type=cfg.vla.trajectory.converter_type,
            trajectory_n_bins=cfg.vla.trajectory.n_bins,
            trajectory_n_dims=cfg.vla.trajectory.n_dims,
        )

        return policy

    # @torch.inference_mode()
    def select_action(self, item: Dict[str, torch.Tensor]) -> torch.Tensor: 
        self.current_step %= 10
        # 每10步，计算一次模型输出，进行纠正一下contorl point
        if self.current_step == 0:
            # 然后就可以利用vla_tokenzier来进行batch的处理，变成模型可以输入的格式 （image0是评测的奇怪不同）
            batch = self.vla_tokenizer.tokenize_input(
                dict(
                    cam1=item[DATASET_ITEM_MAP_KEYS[self.dataset_name]["cam1"]],
                    cam2=item[DATASET_ITEM_MAP_KEYS[self.dataset_name]["cam2"]],
                    language=item[DATASET_ITEM_MAP_KEYS[self.dataset_name]["language"]][0],
                )
            )
            
            # 统一数据类型到模型的 dtype
            pixel_values, image_transform = (
                batch["pixel_values"],
                self.model.vision_backbone.image_transform,
            )

            pixel_values["cam1"] = (
                image_transform(pixel_values["cam1"]).unsqueeze(0).to(self.model.device)
            )
            pixel_values["cam2"] = (
                image_transform(pixel_values["cam2"]).unsqueeze(0).to(self.model.device)
            )

            input_ids = (
                torch.tensor(batch["prompt_ids"], dtype=torch.long)
                .unsqueeze(0)
                .to(self.model.device)
            )
            print("input_ids:", input_ids)

            with torch.autocast("cuda", dtype=self.model.llm_backbone.half_precision_dtype):
                pred_ids = GenerationMixin.generate(
                    self.model,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    use_cache=False,
                    do_sample=False,  # 强制贪心解码
                    max_new_tokens=1024,  
                )
            pred_ids = pred_ids.cpu().numpy()
            print("shape:", pred_ids.shape, "pred_ids: ", pred_ids)
            input_ids_len = input_ids.shape[1]
            pred_action_ids = pred_ids[0, input_ids_len:] # 一个batch的情况
            
            decoded_control_points = self.trajectory_converter.decode_text_ids_to_trajectory(pred_action_ids)
            # TODO： 进行优化修正control point，拿到优化过的控制点。

            # ========================检测并删除knot不连续的异常点 =====
            knot_times = decoded_control_points[:, -1]
            abnormal_indices = []
            
            for i in range(1, len(knot_times)):
                if knot_times[i] <= knot_times[i-1]:  # 如果不严格递增，就是异常
                    abnormal_indices.append(i)
            
            if abnormal_indices:
                print(f"⚠️ 检测到 {len(abnormal_indices)} 个异常control point (knot不连续)")
                for idx in abnormal_indices:
                    print(f"\n异常位置: index={idx}")
                    print(f"前一个点 (index={idx-1}): knot_time={knot_times[idx-1]}, data={decoded_control_points[idx-1]}")
                    print(f"异常点 (index={idx}): knot_time={knot_times[idx]}, data={decoded_control_points[idx]}")
                    if idx + 1 < len(decoded_control_points):
                        print(f"后一个点 (index={idx+1}): knot_time={knot_times[idx+1]}, data={decoded_control_points[idx+1]}")
                
                # 删除异常点
                decoded_control_points = np.delete(decoded_control_points, abnormal_indices, axis=0)
                print(f"\n✅ 已删除 {len(abnormal_indices)} 个异常点，剩余 {len(decoded_control_points)} 个点")
            else:
                print("✓ 所有control points knot连续，无异常点")

            # 如果控制点数量小于3,就直接重复最后的控制点补足到3个
            if len(decoded_control_points) < 3:
                n_missing = 3 - len(decoded_control_points)
                if len(decoded_control_points) == 0:
                    raise ValueError("decoded_control_points 为空，无法补足控制点！")
                last_cp = decoded_control_points[-1:]
                decoded_control_points = np.concatenate([
                    decoded_control_points,
                    np.repeat(last_cp, n_missing, axis=0)
                ], axis=0)
                print(f"⚠️ 控制点数量不足3，已重复最后一个控制点补足，当前shape: {decoded_control_points.shape}")
            # =========================额外异常处理结束====================

            if self.decode_control_point is None:
                self.decode_control_point = decoded_control_points
            else: # 进行平滑化，不过现在就直接更新测试一下，会不会有改善
                self.decode_control_point = decoded_control_points

            np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
            print("decoded_control_points:",
                  decoded_control_points)

            next_action, bspline = self.trajectory_compression.decode_to_action(
                self.decode_control_point, current_eef_pose=self.last_action
            )

            # 重建整条轨迹：用bspline在knot时间点上采样
            knot_times = self.decode_control_point[:, -1]
            # 采样数量为 knot_times[-1] 的整数部分，从 0 到 knot_times[-1] -1 （不算最后一个点）
            self.last_step = int(knot_times[-1]) 
            t_eval = np.arange(0, self.last_step, 0.2)
            reconstructed_traj = np.zeros((len(t_eval), 7))
            reconstructed_traj[:, :6] = bspline(t_eval)
            # 变成realtive, NOTE：起始点不是00000,只能用relative来操作
            reconstructed_traj[:-1, :6] = np.diff(reconstructed_traj[:, :6], axis=0)
            reconstructed_traj[-1] = np.array([0, 0, 0, 0, 0, 0, -1])
            # gripper用线性插值 reconstructed_traj[:, 6] = np.interp(t_eval, knot_times, decoded_control_points[:, 6])
            # gripper采用0阶的插值方式
            indices = np.searchsorted(knot_times, t_eval, side='right') - 1
            reconstructed_traj[:, 6] = self.decode_control_point[indices, 6]
            
            self.reconstruct_traj = reconstructed_traj
            print("reconstructed traj:", self.reconstruct_traj)

        action = self.reconstruct_traj[self.current_step]
        print("self.current_step:", self.current_step, " .conduct action:", action)

        self.current_step += 1
        # self.current_step = min(self.current_step, self.last_step-1)
        self.last_action = action
        return torch.Tensor(action).unsqueeze(0)

    def predict_action_chunk(
        self,
        observation: Dict[str, Union[torch.Tensor, np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict action chunk (sequence of actions).
        NOTE: action horizon这个东西让模型自己去处理
        Used by action-chunking policies to predict multiple future steps.

        Args:
            observation: Dict with "image" and "task"

        Returns:
            torch.Tensor of shape [B, action_horizon, action_dim]
        """
        print("进入predict action chunk的地方")
        # For VLA, we typically generate one step at a time
        # But we can repeat to create a chunk
        action = self.select_action(observation, **kwargs)

        return action

    def forward(self, batch: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def get_optim_params(self) -> list:
        pass
    # ============ Standard nn.Module methods ============

    def to(self, device: Union[str, torch.device], **kwargs) -> "VLAPolicy":
        """Move policy to device."""
        self.model = self.model.to(device, **kwargs)
        return self

    def eval(self) -> "VLAPolicy":
        """Set to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "VLAPolicy":
        """Set to training mode."""
        self.model.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.model.parameters()).device


if __name__ == "__main__":
    # 这里是如果是一个测试，用来从lerobot数据集读取的图像，来给这个policy进行测试，看输出是否正常。
    # 以及可以对比一下如果用训练的方式，是否是正常的。
    policy = VLAPolicy.from_pretrained(
        pretrained_name_or_path="/inspire/hdd/project/robot-decision/hexinyu-253108100063/Project/Aff/vla_runs/"
        + "base+b64+x7--2_train2eval_without_flashattn/"
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        policy.dataset_name,
        root=Path(
            f"/inspire/hdd/project/robot-decision/public/datasets/{policy.dataset_name}"
        ),
    )

    from torch.utils.data import DataLoader

    base_tokenizer = policy.model.llm_backbone.tokenizer
    # from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    # collator = PaddedCollatorForActionPrediction(
    #     base_tokenizer.model_max_length, base_tokenizer.pad_token_id
    # )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False
    )  # , collate_fn=collator)

    for idx, batch in enumerate(dataloader):
        print(f"===== Sample {idx} =====")
        action = policy.select_action(batch)
        print("policy测试输出action:", action)

        break
