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

from .configuration_vla import VLAConfig

logger = logging.getLogger(__name__)


class VLAPolicy(PreTrainedPolicy):
    """
    VLA Policy compatible with LeRobot evaluation framework.

    This policy wraps a VLA model (Vision-Language-Action) and provides:
    - Standard PreTrainedPolicy interface for LeRobot
    - Batched inference support
    - State management for recurrent components
    - Checkpoint loading and saving

    Key methods:
    - from_pretrained: Load from checkpoint
    - select_action: Single forward pass for rollouts
    - predict_action_chunk: Predict action sequence (for action chunking)
    - forward: Training forward pass
    """

    config_class = VLAConfig
    name = "vla"

    def __init__(
        self,
        config: VLAConfig,
        **kwargs,
    ):
        """
        Initialize VLA Policy.

        Args:
            config: VLAConfig instance
            **kwargs: Additional arguments passed to PreTrainedPolicy
        """
        super().__init__(config)
        self.config = config

        # Initialize model
        self.model = self._build_model()

        # State for recurrent models
        self._hidden_state = None
        self._cache = {}

    def _build_model(self) -> nn.Module:
        """Build the VLA model from config."""
        try:
            from prismatic.models.vlms import PrismaticVLM

            model = PrismaticVLM(
                vision_backbone_id=self.config.vision_backbone,
                llm_backbone_id=self.config.llm_backbone,
            )
            return model
        except ImportError as e:
            logger.error(f"Failed to import PrismaticVLM: {e}")
            raise

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        config: Optional[VLAConfig] = None,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "VLAPolicy":
        """
        Load VLA policy from checkpoint.

        Args:
            pretrained_model_name_or_path: Path to checkpoint directory
            config: Optional VLAConfig (loaded from config.json if not provided)
            device: Device to load model on
            dtype: Data type for model weights
            **kwargs: Additional arguments

        Returns:
            VLAPolicy instance
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        # Load config from checkpoint if not provided
        if config is None:
            config_path = pretrained_model_name_or_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")

            with open(config_path, "r") as f:
                config_dict = json.load(f)

            config = VLAConfig(**config_dict)
            logger.info(f"Loaded config from {config_path}")

        # Create policy instance
        policy = cls(config, **kwargs)

        # Load model weights
        model_path = pretrained_model_name_or_path / "model.safetensors"
        if not model_path.exists():
            model_path = pretrained_model_name_or_path / "model.pt"

        if model_path.exists():
            logger.info(f"Loading model weights from {model_path}")
            if str(model_path).endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(model_path)
            else:
                checkpoint = torch.load(model_path, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint)

            policy.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"No model weights found in {pretrained_model_name_or_path}")

        # Move to device and set dtype
        if device and device != "cpu":
            policy.model = policy.model.to(device)
        if dtype is not None:
            policy.model = policy.model.to(dtype=dtype)

        policy.model.eval()
        return policy

    def select_action(
        self,
        observation: Dict[str, Union[torch.Tensor, np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Select single action from observation.

        This method is called at each environment step during rollout.

        Args:
            observation: Dict containing:
                - "image": [B, C, H, W] tensor/array (RGB images)
                - "task": str or list[str] (task description)
                - ... other keys

        Returns:
            torch.Tensor of shape [B, action_dim]
        """
        # Convert numpy to tensor if needed
        if isinstance(observation, dict):
            observation = {
                k: (
                    torch.as_tensor(v, device=self.device)
                    if isinstance(v, np.ndarray)
                    else v
                )
                for k, v in observation.items()
            }

        image = observation.get("image")
        task = observation.get("task", "")

        if image is None:
            raise ValueError("Observation must contain 'image' key")

        batch_size = image.shape[0] if image.dim() > 0 else 1

        # Handle task description
        if isinstance(task, str):
            task = [task] * batch_size

        with torch.no_grad():
            try:
                # Forward pass through model
                if hasattr(self.model, "select_action"):
                    action = self.model.select_action(
                        images=image,
                        task_descriptions=task,
                    )
                else:
                    # Fallback: try forward pass
                    action = self.model(
                        pixel_values=image,
                        language=task,
                    )

                # Ensure correct shape and device
                if not isinstance(action, torch.Tensor):
                    action = torch.as_tensor(action, device=self.device)

                action = action.to(device=self.device, dtype=torch.float32)

                # Ensure [B, action_dim] shape
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                if action.shape[0] != batch_size:
                    action = action[:batch_size]

            except Exception as e:
                logger.warning(f"Forward pass failed: {e}. Returning zero action.")
                action = torch.zeros(
                    batch_size,
                    self.config.action_dim,
                    device=self.device,
                    dtype=torch.float32,
                )

        return action

    def predict_action_chunk(
        self,
        observation: Dict[str, Union[torch.Tensor, np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict action chunk (sequence of actions).

        Used by action-chunking policies to predict multiple future steps.

        Args:
            observation: Dict with "image" and "task"

        Returns:
            torch.Tensor of shape [B, action_horizon, action_dim]
        """
        # For VLA, we typically generate one step at a time
        # But we can repeat to create a chunk
        action = self.select_action(observation, **kwargs)

        # Repeat for action_horizon
        if action.dim() == 2:  # [B, action_dim]
            action = action.unsqueeze(1)  # [B, 1, action_dim]
            action = action.repeat(
                1, self.config.action_horizon, 1
            )  # [B, horizon, action_dim]

        return action

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for training/validation.

        Args:
            batch: Dict containing:
                - "observation": observation dict
                - "action": [B, seq_len, action_dim] ground truth actions
                - "action_is_pad": [B, seq_len] padding mask (optional)

        Returns:
            (loss, loss_dict) where loss is scalar and loss_dict contains logging info
        """
        observation = batch.get("observation", {})
        gt_action = batch.get("action")

        if gt_action is None:
            raise ValueError("Batch must contain 'action' key for training")

        # Get predicted action
        pred_action = self.select_action(observation)

        # Compute loss (simple MSE for now)
        loss = torch.nn.functional.mse_loss(pred_action, gt_action[:, 0, :])

        loss_dict = {
            "mse_loss": loss.item(),
            "mae_loss": torch.nn.functional.l1_loss(
                pred_action, gt_action[:, 0, :]
            ).item(),
        }

        return loss, loss_dict

    def reset(self) -> None:
        """
        Reset state (called when environment resets).

        Used for clearing hidden states, caches, etc.
        """
        self._hidden_state = None
        self._cache = {}

        if hasattr(self.model, "reset"):
            self.model.reset()

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
