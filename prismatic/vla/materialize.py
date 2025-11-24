"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""
from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

from prismatic.vla.tokenizer import TRAJECTORY_CONVERTER_REGISTRY, BaseTrajectoryConverter, VlaTokenizer
from prismatic.vla.dataset import MyLeRobotDataset
from prismatic.vla.trajectory_compression import TRAJECTORY_COMPRESSION_REGISTRY

def get_vla_dataset_and_collator(
        data_repo_id: str,
        data_task_ids: list[int],
        # Trajectory compression 
        trajectory_compression_method: str,
        # VlaTokenizer构造参数 (required parameters)
        base_tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        image_transform: ImageTransform,
        # Trajectory converter configuration (with defaults)
        trajectory_converter_type: str = 'value_textualize',
        trajectory_n_bins: int = 256,
        trajectory_n_dims: int = 7,
        # Optional parameters
        predict_stop_token: bool = True,
        default_image_resolution: Tuple[int, int] = (224, 224),
        # 给padding的参数
        padding_side: str = "right",
    ) -> Tuple[Dataset, BaseTrajectoryConverter, PaddedCollatorForActionPrediction]:
    """新的LeRobot的实现方式"""
    trajectory_converter = TRAJECTORY_CONVERTER_REGISTRY[trajectory_converter_type](
        tokenizer=base_tokenizer, 
        n_bins=trajectory_n_bins,
        n_dims=trajectory_n_dims,
        )
    
    vla_tokenizer = VlaTokenizer(
        trajectory_converter=trajectory_converter,
        base_tokenizer=base_tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        default_image_resolution=default_image_resolution,
    )

    trajectory_compressor = TRAJECTORY_COMPRESSION_REGISTRY[trajectory_compression_method]() # 注意需要括号才能实例化
    dataset = MyLeRobotDataset(
        repo_id=data_repo_id,
        tokenizer=vla_tokenizer,
        trajectory_compression=trajectory_compressor,
        task_ids=data_task_ids
    )

    # Common collator for both dataset types
    collator = PaddedCollatorForActionPrediction(
        base_tokenizer.model_max_length, base_tokenizer.pad_token_id, padding_side=padding_side
    )

    return dataset, trajectory_converter, collator

