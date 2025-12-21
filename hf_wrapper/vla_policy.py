"""
VLA Policy wrapper implementing HuggingFace PreTrainedPolicy interface.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class VLAPolicy(PreTrainedPolicy):
    """
    VLA Policy implementation compatible with HuggingFace's PreTrainedPolicy interface.
    
    This policy wraps the OpenVLA model and provides:
    - Standard HF config/model loading from checkpoint
    - Batched inference support
    - Integration with lerobot's preprocessor/postprocessor pipeline
    - State management for recurrent models
    """

    config_class = AutoConfig
    pretrained_model_name_or_path_required = False

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize VLAPolicy.
        
        Args:
            model: The VLA model (nn.Module)
            config: Configuration dict containing model and task info
            **kwargs: Additional arguments passed to PreTrainedPolicy
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.config = config or {}
        
        # Set model to eval mode by default
        self.model.eval()
        
        # Track whether model has been put on device
        self._device_set = False
        
        # State management for recurrent models (if needed)
        self._hidden_state = None
        self._cache = {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "VLAPolicy":
        """
        Load VLA policy from a checkpoint directory or HuggingFace model identifier.
        
        Args:
            pretrained_model_name_or_path: Path to checkpoint or HF model ID
            device: Device to load model on (cpu, cuda, etc.)
            dtype: Data type for model (torch.float32, torch.bfloat16, etc.)
            **kwargs: Additional arguments
            
        Returns:
            VLAPolicy instance
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        
        # Load config from checkpoint
        config_path = pretrained_model_name_or_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        logger.info(f"Loaded config from {config_path}")
        
        # Load model checkpoint
        model_path = pretrained_model_name_or_path / "model.safetensors"
        if not model_path.exists():
            model_path = pretrained_model_name_or_path / "latest-checkpoint.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {pretrained_model_name_or_path}")
        
        # Import OpenVLA model here to avoid circular imports
        from prismatic.models.vlms import PrismaticVLM
        from prismatic.vla.tokenizer import VlaTokenizer
        from safetensors import safe_open
        from collections import OrderedDict
        
        # Reconstruct model from config and checkpoint
        logger.info("Constructing VLA model...")
        
        # Create base VLM model
        model_config = config.get("model_config", {})
        vlm_model = PrismaticVLM(
            vision_backbone_id=model_config.get("vision_backbone_id", "siglip-vit-so400m-14-384"),
            llm_backbone_id=model_config.get("llm_backbone_id", "distilgpt2"),
        )
        
        # Load checkpoint from safetensors
        with safe_open(str(model_path), framework="pt", device="cpu") as f:
            flat_checkpoint = {k: f.get_tensor(k) for k in f.keys()}
        
        # Reconstruct nested structure from flat keys
        checkpoint = {}
        for key, tensor in flat_checkpoint.items():
            parts = key.split(".")
            current = checkpoint
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = tensor
        
        vlm_model.load_state_dict(checkpoint)
        logger.info(f"Loaded model checkpoint from {model_path}")
        
        # Move to device
        if device != "cpu":
            vlm_model = vlm_model.to(device)
        if dtype is not None:
            vlm_model = vlm_model.to(dtype=dtype)
        
        vlm_model.eval()
        
        # Create policy instance
        policy = cls(
            model=vlm_model,
            config=config,
            device=device,
            dtype=dtype,
        )
        
        return policy

    def select_action(
        self,
        observation: Union[Dict[str, torch.Tensor], Dict[str, np.ndarray]],
        deterministic: bool = True,
        return_action_info: bool = False,
    ) -> Union[PolicyAction, tuple]:
        """
        Select action from observation using the VLA model.
        
        Args:
            observation: Dict containing observation data
                - "image": [B, C, H, W] tensor or numpy array
                - "task": task description string (or list of strings for batch)
            deterministic: Whether to sample deterministically (greedy)
            return_action_info: Whether to return additional action info
            
        Returns:
            PolicyAction: [B, action_dim] tensor of actions
            or tuple of (actions, action_info) if return_action_info=True
        """
        # Handle numpy inputs
        if isinstance(observation, dict):
            observation = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v 
                          for k, v in observation.items()}
        
        # Ensure tensors are on correct device
        for key in observation:
            if isinstance(observation[key], torch.Tensor):
                observation[key] = observation[key].to(self.device)
        
        # Get batch size
        if "image" in observation:
            batch_size = observation["image"].shape[0]
        else:
            batch_size = 1
        
        # Handle task description
        task = observation.get("task", "")
        if isinstance(task, str):
            task = [task] * batch_size
        
        with torch.inference_mode():
            # Prepare model input
            # In a full implementation, you would:
            # 1. Tokenize task description
            # 2. Process image through vision encoder
            # 3. Generate action tokens autoregressively
            # 4. Decode tokens to continuous actions
            
            # For now, provide a skeleton that shows the interface
            image = observation.get("image")  # [B, C, H, W]
            
            if image is None:
                raise ValueError("Observation must contain 'image' key")
            
            # Forward pass through model
            # This is model-specific and would need to be implemented based on
            # your model's forward signature
            try:
                action_output = self.model.select_action(
                    images=image,
                    task_descriptions=task,
                    return_dict=True,
                )
            except Exception as e:
                logger.warning(f"Model forward pass failed: {e}. Returning zero actions.")
                action_output = self._get_zero_action(batch_size)
            
            # Extract action from output
            if isinstance(action_output, dict):
                action = action_output.get("action", action_output.get("output"))
            else:
                action = action_output
            
            # Ensure correct shape and type
            if action is None:
                action = self._get_zero_action(batch_size)
            
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            
            # Ensure [B, action_dim] shape
            if action.ndim == 1:
                action = action.unsqueeze(0)
            
            if action.shape[0] != batch_size:
                logger.warning(f"Action batch size mismatch: {action.shape[0]} vs {batch_size}")
                action = action[:batch_size]
        
        if return_action_info:
            action_info = {
                "deterministic": deterministic,
                "batch_size": batch_size,
            }
            return action, action_info
        
        return action

    def _get_zero_action(self, batch_size: int) -> torch.Tensor:
        """Generate zero action as fallback."""
        action_dim = 7  # x, y, z, roll, pitch, yaw, gripper
        return torch.zeros(batch_size, action_dim, device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        """Reset any state (e.g., for recurrent models)."""
        self._hidden_state = None
        self._cache = {}
        if hasattr(self.model, "reset"):
            self.model.reset()

    def to(self, device: Union[str, torch.device], **kwargs) -> "VLAPolicy":
        """Move policy to device."""
        super().to(device, **kwargs)
        self.model = self.model.to(device, **kwargs)
        self._device_set = True
        return self

    def eval(self) -> "VLAPolicy":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "VLAPolicy":
        """Set model to training mode."""
        self.model.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device if next(self.model.parameters(), None) is not None else torch.device("cpu")

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save policy to directory in HuggingFace format.
        
        Args:
            save_directory: Directory to save to
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save model
        model_path = save_directory / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        logger.info(f"Policy saved to {save_directory}")
