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
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, CustomTrajectoryDataset, CustomRLDSBatchTransform
from prismatic.vla.pose_tokenizer import PoseTokenizer


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    
    # Check if this is a custom trajectory dataset
    if data_mix == "custom_trajectory":
        # Use PoseTokenizer for custom trajectory dataset instead of ActionTokenizer
        pose_tokenizer = PoseTokenizer(tokenizer=tokenizer, bins=256)
        batch_transform = CustomRLDSBatchTransform(
            pose_tokenizer=pose_tokenizer,
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            predict_stop_token=predict_stop_token
        )
        
        # Create custom trajectory dataset
        dataset = CustomTrajectoryDataset(
            data_root_dir, 
            batch_transform, 
            train=train
        )
        
        # For compatibility with training pipeline, return pose_tokenizer as action_tokenizer
        # (They have similar interface for tokenization/de-tokenization)
        action_tokenizer = pose_tokenizer  
        
    else:
        # Standard RLDS dataset path
        action_tokenizer = ActionTokenizer(tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
        )
        
        # Build RLDS Iterable Dataset
        cls = RLDSDataset if not episodic else EpisodicRLDSDataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
        )
    
    # Common collator for both dataset types
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    return dataset, action_tokenizer, collator
