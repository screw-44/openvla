from pathlib import Path
from typing import Tuple, Type
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

from prismatic.vla.tokenizer import (
    TRAJECTORY_CONVERTER_REGISTRY,
    BaseTrajectoryConverter,
    VlaTokenizer,
)
from prismatic.vla.dataset import MyLeRobotDataset
from prismatic.vla.trajectory_compression import TRAJECTORY_COMPRESSION_REGISTRY


def get_vla_tokenizer(
    # Trajectory compression
    trajectory_compression_method: str,
    # VlaTokenizer构造参数 (required parameters)
    base_tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    # Trajectory converter configuration (with defaults)
    trajectory_converter_type: str = "value_textualize",
    trajectory_n_bins: int = 256,
    trajectory_n_dims: int = 7,
):
    trajectory_converter = TRAJECTORY_CONVERTER_REGISTRY[trajectory_converter_type](
        tokenizer=base_tokenizer,
        n_bins=trajectory_n_bins,
        n_dims=trajectory_n_dims,
    )

    vla_tokenizer = VlaTokenizer(
        trajectory_converter=trajectory_converter,
        base_tokenizer=base_tokenizer,
        prompt_builder_fn=prompt_builder_fn,
    )

    trajectory_compressor = TRAJECTORY_COMPRESSION_REGISTRY[
        trajectory_compression_method
    ]()  # 注意需要括号才能实例化

    return trajectory_converter, vla_tokenizer, trajectory_compressor


def get_vla_dataset(
    data_repo_id: str,
    data_task_ids: list[int],
    # Trajectory compression
    trajectory_compression_method: str,
    # VlaTokenizer构造参数 (required parameters)
    base_tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    image_transform: ImageTransform,
    # Trajectory converter configuration (with defaults)
    trajectory_converter_type: str = "value_textualize",
    trajectory_n_bins: int = 256,
    trajectory_n_dims: int = 7,
) -> Tuple[Dataset, BaseTrajectoryConverter, PaddedCollatorForActionPrediction]:

    trajectory_converter, vla_tokenizer, trajectory_compressor = get_vla_tokenizer(
        trajectory_compression_method=trajectory_compression_method,
        base_tokenizer=base_tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        trajectory_converter_type=trajectory_converter_type,
        trajectory_n_bins=trajectory_n_bins,
        trajectory_n_dims=trajectory_n_dims,
    )

    dataset = MyLeRobotDataset(
        repo_id=data_repo_id,
        image_transform=image_transform,
        tokenizer=vla_tokenizer,
        trajectory_compression=trajectory_compressor,
        task_ids=data_task_ids,
    )

    # Common collator for both dataset types
    collator = PaddedCollatorForActionPrediction(
        base_tokenizer.model_max_length, base_tokenizer.pad_token_id
    )

    return dataset, trajectory_converter, collator
