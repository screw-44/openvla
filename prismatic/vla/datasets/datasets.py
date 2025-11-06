"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Generator
import json

import numpy as np
import torch

import tensorflow as tf
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.overwatch import initialize_overwatch

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.pose_tokenizer import PoseTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# Initialize overwatch for logging
overwatch = initialize_overwatch(__name__)


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
    
@dataclass
class CustomRLDSBatchTransform:
    pose_tokenizer: PoseTokenizer  # Will be PoseTokenizer, using Any to avoid import issues
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a custom RLDS batch with 26 9D poses (1 object + 25 trajectory points)."""
        # Extract data from our custom format
        # rlds_batch keys: dict_keys(['rgb', 'language', 'object_pose', 'affordance_trajectory', 'dataset_name'])
        dataset_name = rlds_batch.get("dataset_name", "custom_dataset")
        img = Image.fromarray(rlds_batch['rgb'])
        lang = rlds_batch["language"].lower().strip()
        
        # Extract 9D object pose and 25-step trajectory (each step is 9D)
        object_pose = np.array(rlds_batch["object_pose"], dtype=np.float32)  # Shape: (9,)
        affordance_trajectory = np.array(rlds_batch["affordance_trajectory"], dtype=np.float32)  # Shape: (25, 9)
        # Combine object pose and trajectory into 26 poses
        all_poses = np.vstack([object_pose.reshape(1, 9), affordance_trajectory])  # Shape: (26, 9)
        # Tokenize all 26 poses using pose tokenizer
        pose_tokens_str = self.pose_tokenizer(all_poses)
        
        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"{lang}"},
            {"from": "gpt", "value": pose_tokens_str},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted pose tokens!
        # the expected number of pose tokens: 26 poses × 9 dimensions = 234 tokens
        pose_token_ids = self.base_tokenizer(pose_tokens_str, add_special_tokens=False).input_ids
        
        labels[: -(len(pose_token_ids) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # print("labels's shape:", labels.shape)
        # print("expected pose token ids length:", len(pose_token_ids))
        # exit()

        return dict(
            pixel_values=pixel_values, 
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name
        )


@dataclass
class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class CustomTrajectoryDataset(IterableDataset):
    """Custom dataset for trajectory data with RGB, language, and affordance trajectory information from RLDS format."""
    
    def __init__(
        self,
        data_root_dir: Path, # /home/hxy/Desktop/Affordance_dataset_generation/convert_dataset/ob_hand_hold_rlds
        batch_transform: CustomRLDSBatchTransform,
        train: bool = True,
        shuffle_buffer_size: int = 100,  # Number of episodes to buffer for shuffling
    ) -> None:
        """
        Initialize custom trajectory dataset that reads from RLDS format data.
        
        Args:
            data_root_dir: Path to the RLDS dataset directory (e.g., ob_hand_hold_rlds)
            batch_transform: CustomRLDSBatchTransform instance
            train: Whether this is for training or evaluation
            shuffle_buffer_size: Size of shuffle buffer (only used if train=True)
        """
        self.data_root_dir = data_root_dir
        self.batch_transform = batch_transform
        self.train = train
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Load episodes from RLDS dataset
        self.episodes = self._load_episodes_from_rlds()
        
        # Dataset statistics for action normalization - load from saved stats or compute
        self.dataset_statistics = self._compute_dataset_statistics()
    
    def _load_episodes_from_rlds(self):
        """Load episodes from RLDS TFRecord format."""
        episodes = []
        overwatch.info(f"Loading RLDS dataset from: {self.data_root_dir}")
        
        # Find TFRecord files in the data directory
        tfrecord_files = list(self.data_root_dir.glob("*.tfrecord"))
        if not tfrecord_files:
            overwatch.error(f"No TFRecord files found in {self.data_root_dir}")
            return []
        
        overwatch.info(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Load dataset info if available
        dataset_info_path = self.data_root_dir / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            overwatch.info(f"Dataset contains {dataset_info.get('total_episodes', 'unknown')} episodes")
        
        # Read TFRecord files
        for tfrecord_file in tfrecord_files:
            overwatch.info(f"Processing {tfrecord_file}")
            
            # Create TFRecord dataset
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            
            for i, serialized_example in enumerate(dataset):
                try:
                    # Parse the TFRecord
                    episode_data = self._parse_tfrecord_episode(serialized_example)
                    if episode_data is not None:
                        episodes.append(episode_data)
                        
                except Exception as e:
                    overwatch.warning(f"Failed to parse episode {i}: {e}")
                    continue
        
        overwatch.info(f"Successfully loaded {len(episodes)} episodes")
        return episodes


    
    def _parse_tfrecord_episode(self, serialized_example):
        """Parse a serialized TFRecord example into episode data."""  
        # Define the expected feature structure for the RLDS format
        feature_description = {
            # Episode metadata
            'episode_metadata/episode_id': tf.io.FixedLenFeature([], tf.int64),
            'episode_metadata/object_name': tf.io.FixedLenFeature([], tf.string),
            'episode_metadata/language_prompt': tf.io.FixedLenFeature([], tf.string),
            'episode_metadata/affordance_key': tf.io.FixedLenFeature([], tf.string),
            
            # Observation data
            'observation/rgb': tf.io.FixedLenFeature([], tf.string),  # JPEG encoded
            
            # Pose data
            'object_pose': tf.io.FixedLenSequenceFeature([9], tf.float32, allow_missing=True),
            'affordance': tf.io.FixedLenSequenceFeature([9], tf.float32, allow_missing=True),
        }
        
        # Parse the example
        parsed_features = tf.io.parse_single_example(serialized_example,feature_description)
        
        # Decode RGB image
        rgb_encoded = parsed_features['observation/rgb']
        rgb = tf.io.decode_jpeg(rgb_encoded, channels=3)
        rgb = tf.image.resize(rgb, [224, 224])  # Resize to standard size
        rgb_np = rgb.numpy().astype(np.uint8)
        
        # Extract other data
        episode_id = int(parsed_features['episode_metadata/episode_id'].numpy())
        object_name = parsed_features['episode_metadata/object_name'].numpy().decode('utf-8')
        language = parsed_features['episode_metadata/language_prompt'].numpy().decode('utf-8')
        affordance_key = parsed_features['episode_metadata/affordance_key'].numpy().decode('utf-8')
        
        # Extract poses
        object_pose = parsed_features['object_pose'].numpy()
        affordance_trajectory = parsed_features['affordance'].numpy()
        
        # Ensure proper shapes - object_pose should be (9,)
        object_pose = object_pose.flatten()[:9].astype(np.float32)
        
        # Ensure trajectory is (25, 9)
        affordance_trajectory = self._ensure_trajectory_shape(affordance_trajectory)
        
        episode_data = {
            'rgb': rgb_np,
            'language': language,
            'object_pose': object_pose,
            'affordance_trajectory': affordance_trajectory,
            'dataset_name': 'custom_trajectory_dataset',
            'metadata': {
                'episode_id': episode_id,
                'object_name': object_name,
                'affordance_key': affordance_key,
            }
        }
        
        return episode_data
            
    
    def _ensure_trajectory_shape(self, trajectory):
        """Ensure trajectory is exactly (25, 9) shape."""
        trajectory = np.array(trajectory, dtype=np.float32)
        
        # If 1D, reshape to (N, 9) assuming it's flattened
        if len(trajectory.shape) == 1:
            trajectory = trajectory.reshape(-1, 9)
        
        # If we have more or less than 25 steps, fix it
        if trajectory.shape[0] != 25:
            # Create new trajectory with exactly 25 steps
            new_trajectory = np.zeros((25, 9), dtype=np.float32)
            copy_steps = min(25, trajectory.shape[0])
            new_trajectory[:copy_steps] = trajectory[:copy_steps]
            
            # If we have less than 25 steps, repeat the last valid step
            if copy_steps < 25 and copy_steps > 0:
                new_trajectory[copy_steps:] = trajectory[copy_steps-1]
            
            trajectory = new_trajectory
        
        return trajectory
    
    def _compute_dataset_statistics(self):
        """Load saved dataset statistics or compute from data."""
        return {
            "custom_trajectory_dataset": {
                "action": {
                    "q01": np.array([-1.0] * 9, dtype=np.float32),  # 9 维
                    "q99": np.array([3.4] * 9, dtype=np.float32)     # 9 维
                }
            }
    }
    
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate over episodes and yield transformed data.
        For training, this will loop infinitely with shuffling.
        For evaluation, this will iterate once through the dataset.
        """
        import random
        
        while True:  # Infinite loop for training
            # Create a copy of episode indices
            episode_indices = list(range(len(self.episodes)))
            
            # Shuffle if in training mode
            if self.train:
                random.shuffle(episode_indices)
            
            # Iterate through episodes
            for idx in episode_indices:
                episode_data = self.episodes[idx]
                # Apply the CustomRLDSBatchTransform and yield
                yield self.batch_transform(episode_data)
            
    
    def __len__(self) -> int:
        """Return number of episodes in the dataset."""
        return len(self.episodes)
    