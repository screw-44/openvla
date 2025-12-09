"""
policy_wrapper.py

Wraps OpenVLA model to provide LeRobot-compatible interface for evaluation.
Does NOT modify lerobot library or existing VLA implementation.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from PIL import Image

from prismatic.conf.vla import VLAConfig
from prismatic.models import load
from prismatic.overwatch import initialize_overwatch
from .processor_factory import make_openvla_processors

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


class OpenVLAPolicyWrapper:
    """
    Wrapper around OpenVLA model to provide LeRobot-compatible interface.
    
    Key Features:
    - Loads VLA model using unified load() function
    - Provides select_action() interface for lerobot_eval
    - Uses processor pipeline to handle input/output transformations
    - Maintains compatibility with existing VLA training code
    
    Usage:
        policy = OpenVLAPolicyWrapper.from_pretrained("path/to/checkpoint")
        action = policy.select_action(observation_dict)
    """
    
    def __init__(
        self,
        checkpoint_path: Path
    ):
        """
        Initialize the policy wrapper.
        
        Args:
            checkpoint_path: Path to VLA checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load configuration
        config = self._load_config()
        vla_cfg = self._create_vla_config(config)
        
        # Load VLA model using unified load function with new API
        overwatch.info(f"Loading VLA model from: {self.checkpoint_path}")
        self.vla = load(
            vla_cfg=vla_cfg,
            checkpoint_path=str(self.checkpoint_path),
            load_for_training=False,  # Inference mode
        )
        # Initialize processors
        overwatch.info("Initializing processor pipelines...")
        self.preprocessor, self.postprocessor = make_openvla_processors(
            vla_model=self.vla,
            config_dict=config,
            dataset_ref=None,  # Will be set if needed for trajectory retrieval
        )
        
        # Keep reference to important components
        self.trajectory_converter = self.vla.trajectory_converter
        
        overwatch.info("Policy wrapper initialized successfully!")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json in checkpoint directory."""
        # Checkpoint path should be like: run_dir/checkpoints/step-xxx.pt
        config_path = self.checkpoint_path.parents[1] / "config.json"
        if not config_path.exists(): 
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f: 
            config = json.load(f)
        overwatch.info(f"Loaded config from: {config_path}")
        return config
    
    def _create_vla_config(self, config_dict: Dict[str, Any]) -> VLAConfig:
        """
        Create VLAConfig instance from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary loaded from config.json
        
        Returns:
            VLAConfig instance with required fields populated
        """
        # Extract VLA config from the dict - handle nested or flat structures
        vla_cfg_dict = config_dict["vla"]
        
        # Create VLAConfig with required and optional fields （从json加载，而不是通过vla_id加载。避免修改代码导致无法复现)
        vla_cfg = VLAConfig(
            vla_id=vla_cfg_dict.get("vla_id", "test_policy"),
            base_vlm=vla_cfg_dict.get("base_vlm", "unknown"),
            freeze_vision_backbone=vla_cfg_dict.get("freeze_vision_backbone", False),
            freeze_llm_backbone=vla_cfg_dict.get("freeze_llm_backbone", False),
            unfreeze_last_llm_layer=vla_cfg_dict.get("unfreeze_last_llm_layer", False),
            shuffle_buffer_size=vla_cfg_dict.get("shuffle_buffer_size", 256_000),
            global_batch_size=vla_cfg_dict.get("global_batch_size", 256),
            per_device_batch_size=vla_cfg_dict.get("per_device_batch_size", 32),
            learning_rate=vla_cfg_dict.get("learning_rate", 2e-5),
            weight_decay=vla_cfg_dict.get("weight_decay", 0.0),
            max_grad_norm=vla_cfg_dict.get("max_grad_norm", 1.0),
            lr_scheduler_type=vla_cfg_dict.get("lr_scheduler_type", "constant"),
            warmup_ratio=vla_cfg_dict.get("warmup_ratio", 0.0),
            train_strategy=vla_cfg_dict.get("train_strategy", "fsdp-full-shard"),
            enable_gradient_checkpointing=vla_cfg_dict.get("enable_gradient_checkpointing", True),
            enable_mixed_precision_training=vla_cfg_dict.get("enable_mixed_precision_training", True),
            reduce_in_full_precision=vla_cfg_dict.get("reduce_in_full_precision", True),
            trajectory_converter_type=vla_cfg_dict.get("trajectory_converter_type", "value_textualize"),
            trajectory_n_bins=vla_cfg_dict.get("trajectory_n_bins", 256),
            trajectory_n_dims=vla_cfg_dict.get("trajectory_n_dims", 7),
        )
        
        return vla_cfg
    
    @torch.no_grad()
    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Select action given observation (for lerobot_eval compatibility).
        
        Args:
            observation: Dictionary containing:
                - 'full_image': np.ndarray [H, W, 3] (or similar image key)
                - 'task' or instruction: str
                - Other observation fields as needed
        
        Returns:
            action: np.ndarray [action_dim] - continuous action
        """
        self.vla.eval()
        
        # Preprocess observation
        # Note: Preprocessor expects certain keys, map observation to expected format
        processed = self._preprocess_observation(observation)
        
        # Get image and instruction
        image = processed['image']
        instruction = processed['instruction']
        
        # Use VLA's predict_action method (existing implementation)
        result = self.vla.predict_action(
            image=image,
            instruction=instruction,
        )
        
        # Extract continuous action from result dict
        action = result['action']
        
        return action
    
    def _preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation to format expected by VLA.
        
        Handles different observation formats and extracts:
        - Image (as PIL.Image)
        - Instruction/task string
        - Un-normalization key
        """
        processed = {}
        
        # Extract image
        if 'full_image' in observation:
            img_array = observation['full_image']
        elif 'observation.images.image' in observation:
            img_array = observation['observation.images.image']
        else:
            # Try to find any image key
            img_keys = [k for k in observation.keys() if 'image' in k.lower()]
            if img_keys:
                img_array = observation[img_keys[0]]
            else:
                raise ValueError(f"No image found in observation keys: {observation.keys()}")
        
        # Convert to PIL Image if needed
        if isinstance(img_array, np.ndarray):
            # Handle tensor format [C, H, W] or [H, W, C]
            if img_array.ndim == 3 and img_array.shape[0] == 3:
                img_array = img_array.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
            
            # Ensure uint8 range
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                img_array = (img_array * 255).astype(np.uint8)
            
            processed['image'] = Image.fromarray(img_array)
        elif isinstance(img_array, Image.Image):
            processed['image'] = img_array
        else:
            processed['image'] = Image.fromarray(np.array(img_array))
        
        # Extract instruction/task
        if 'task' in observation:
            processed['instruction'] = observation['task']
        elif 'instruction' in observation:
            processed['instruction'] = observation['instruction']
        else:
            # Default instruction
            processed['instruction'] = "perform the task"
        
        return processed
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
    ) -> "OpenVLAPolicyWrapper":
        """
        Load policy from pretrained checkpoint (LeRobot-compatible interface).
        
        Args:
            checkpoint_path: Path to checkpoint file or run directory
        
        Returns:
            Initialized policy wrapper
        """
        checkpoint_path = Path(checkpoint_path)
        
        # If path is a directory, find the latest checkpoint
        if checkpoint_path.is_dir():
            checkpoint_dir = checkpoint_path / "checkpoints"
            if checkpoint_dir.exists():
                # Find latest checkpoint
                checkpoints = list(checkpoint_dir.glob("step-*.pt"))
                if checkpoints:
                    # Sort by step number
                    checkpoints.sort(key=lambda p: int(p.stem.split('-')[1]))
                    checkpoint_path = checkpoints[-1]
                    overwatch.info(f"Found latest checkpoint: {checkpoint_path.name}")
                else:
                    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            else:
                raise FileNotFoundError(f"No checkpoints directory in {checkpoint_path}")
        
        return cls(checkpoint_path=checkpoint_path)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (if needed for compatibility).
        
        Args:
            batch: Batch dictionary with input_ids, pixel_values, labels
        
        Returns:
            Dictionary with loss and other outputs
        """
        output = self.vla(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch["pixel_values"],
            labels=batch.get("labels"),
        )
        
        return {
            "loss": output.loss,
            "logits": output.logits,
        }
    
    def __repr__(self) -> str:
        return f"OpenVLAPolicyWrapper(checkpoint={self.checkpoint_path.name})"
