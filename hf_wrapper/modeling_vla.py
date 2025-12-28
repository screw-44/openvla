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
from prismatic.vla.tokenizer import VlaTokenizer
from prismatic.vla.trajectory_compression import BaseTrajectoryCompression
from prismatic.models.vlms.base_vlm import VLM
from prismatic.vla.dataset import DATASET_ITEM_MAP_KEYS

logger = logging.getLogger(__name__)


class VLAPolicy(PreTrainedPolicy):
    """
    VLA Policy compatible with LeRobot evaluation framework.
    """

    config_class = VLAConfig
    name = "vla"

    def __init__(self, config: VLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model: VLA = None

        self.trajectory_converter = None
        self.vla_tokenizer: VlaTokenizer = None
        self.trajectory_compression: BaseTrajectoryCompression = None

        self.dataset_name = "HuggingFaceVLA/libero"
        self.last_output = None

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

        # Load config from checkpoint if not provided
        if config is None:
            config_path = pretrained_name_or_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")

            # 直接读取本地 JSON 文件，而不用 from_pretrained（它期望 HF Hub repo id）
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VLAConfig(**config_dict)

            logger.info(
                f"Load config from {config_path},type:{config.__class__.__name__}"
            )

        # Create policy instance
        policy = cls(config, **kwargs)

        # Load model weights
        policy.model = load(
            vla_cfg=config,
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
            trajectory_compression_method=config.trajectory_compression,
            base_tokenizer=policy.model.llm_backbone.tokenizer,
            prompt_builder_fn=policy.model.llm_backbone.prompt_builder_fn,
            trajectory_converter_type=config.trajectory_converter_type,
            trajectory_n_bins=config.trajectory_n_bins,
            trajectory_n_dims=config.trajectory_n_dims,
        )

        return policy

    # @torch.inference_mode()
    def select_action(self, item: Dict[str, torch.Tensor]) -> torch.Tensor:

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
        attention_mask = input_ids.ne(
            self.model.llm_backbone.tokenizer.pad_token_id
        )

        with torch.autocast("cuda", dtype=self.model.llm_backbone.half_precision_dtype):
            generated_ids = GenerationMixin.generate(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False,
                do_sample=False,  # 强制贪心解码
                # max_new_tokens=8,  # 手动添加，这是不应该的
            )
        generated_ids = generated_ids[0, input_ids.shape[1] :].cpu()


        if len(generated_ids) != 8:
            return self.last_output

        self.last_output = torch.Tensor(self.model.trajectory_converter.decode_text_ids_to_trajectory(
            generated_ids
        ))
        
        return self.last_output

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
        # For VLA, we typically generate one step at a time
        # But we can repeat to create a chunk
        action = self.select_action(observation, **kwargs)

        return action

    def forward(self, batch: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def get_optim_params(self) -> list:
        """
        Return optimizer parameter groups.

        Can specify different learning rates for different components.
        """
        return [
            {
                "params": self.model.parameters(),
                "lr": self.config.optimizer_lr,
            }
        ]

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

    @staticmethod
    def _get_config_class_by_vla_id(vla_id: str) -> type:
        """
        Get the config class corresponding to a vla_id.

        This is useful when you want to explicitly load a specific subclass
        instead of relying on draccus's automatic selection.

        Args:
            vla_id: The vla_id string (e.g., "base_4090", "base", "distilgpt2")

        Returns:
            The corresponding config class (e.g., Base_4090, Base, DistilGPT2)
        """
        for variant in VLARegistry:
            if variant.value.vla_id == vla_id:
                return variant.value

        raise ValueError(
            f"Unknown vla_id '{vla_id}'. Available options: "
            f"{[v.vla_id for v in VLARegistry]}"
        )


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
