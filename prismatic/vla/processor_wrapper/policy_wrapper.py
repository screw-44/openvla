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
        checkpoint_path: Path,
        config_dict: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the policy wrapper.
        
        Args:
            checkpoint_path: Path to VLA checkpoint
            config_dict: Configuration dictionary (loaded from config.json if None)
            device: Device to run model on
            hf_token: HuggingFace token for model access
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        # Load configuration if not provided
        if config_dict is None:
            config_dict = self._load_config()
        self.config = config_dict
        
        # Load VLA model using unified load function
        overwatch.info(f"Loading VLA model from: {self.checkpoint_path}")
        self.vla = load(
            str(self.checkpoint_path),
            hf_token=hf_token,
            load_for_training=False,  # Inference mode
        )
        
        # Move to device
        if not hasattr(self.vla, 'device'):
            self.vla = self.vla.to(device)
        
        # Initialize processors
        overwatch.info("Initializing processor pipelines...")
        self.preprocessor, self.postprocessor = make_openvla_processors(
            vla_model=self.vla,
            config_dict=self.config,
            dataset_ref=None,  # Will be set if needed for trajectory retrieval
        )
        
        # Keep reference to important components
        self.trajectory_converter = self.vla.trajectory_converter
        
        overwatch.info("Policy wrapper initialized successfully!")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json in checkpoint directory."""
        # Checkpoint path should be like: run_dir/checkpoints/step-xxx.pt
        run_dir = self.checkpoint_path.parents[1]
        config_path = run_dir / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        overwatch.info(f"Loaded config from: {config_path}")
        return config
    
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
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ) -> "OpenVLAPolicyWrapper":
        """
        Load policy from pretrained checkpoint (LeRobot-compatible interface).
        
        Args:
            checkpoint_path: Path to checkpoint file or run directory
            device: Device to run on
            hf_token: HuggingFace token
        
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
        
        return cls(
            checkpoint_path=checkpoint_path,
            config_dict=None,  # Will auto-load
            device=device,
            hf_token=hf_token,
        )
    
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
        return f"OpenVLAPolicyWrapper(checkpoint={self.checkpoint_path.name}, device={self.device})"
