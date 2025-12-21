"""
Processor pipeline for VLA policy following lerobot standards.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline


class VLAPreprocessor(PolicyProcessorPipeline):
    """
    Preprocessor for VLA policy input.
    
    Converts raw observations to model input format:
    - RGB images: numpy [H, W, 3] uint8 -> torch [1, 3, H, W] float32 normalized
    - Task description: string -> kept as string for model
    """

    def __init__(
        self,
        image_size: int = 224,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size
            image_mean: Image normalization mean (ImageNet default if None)
            image_std: Image normalization std (ImageNet default if None)
        """
        super().__init__(**kwargs)
        
        self.image_size = image_size
        
        # Default to ImageNet normalization
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std).view(1, 3, 1, 1)

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process observation dict.
        
        Args:
            observation: Input observation dict with keys:
                - "image": numpy [H, W, 3] uint8 or torch tensor
                - "task": string task description
                
        Returns:
            Processed observation dict ready for model input
        """
        processed = {}
        
        # Process image
        if "image" in observation:
            img = observation["image"]
            
            # Convert to tensor if numpy
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            
            # Handle different input formats
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            
            # Ensure CHW format
            if img.ndim == 3:
                if img.shape[-1] == 3 and img.shape[0] != 3:
                    # HWC -> CHW
                    img = img.permute(2, 0, 1)
                # Add batch dimension if needed
                if img.ndim == 3:
                    img = img.unsqueeze(0)
            
            # Resize if needed
            if img.shape[-2:] != (self.image_size, self.image_size):
                import torch.nn.functional as F
                img = F.interpolate(
                    img, 
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            
            # Normalize
            img = (img - self.image_mean) / self.image_std
            
            processed["image"] = img
        
        # Pass through task description as-is
        if "task" in observation:
            processed["task"] = observation["task"]
        
        return processed


class VLAPostprocessor(PolicyProcessorPipeline):
    """
    Postprocessor for VLA policy output.
    
    Converts model output to environment action format:
    - Model output: [B, action_dim] tensor
    - Environment action: [B, action_dim] numpy array or tensor
    """

    def __call__(self, action_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process model output.
        
        Args:
            action_output: Model output dict with key "action"
            
        Returns:
            Dict with processed action
        """
        if isinstance(action_output, dict):
            action = action_output.get("action", action_output)
        else:
            action = action_output
        
        # Convert to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        
        # Ensure [B, action_dim] shape
        if action.ndim == 1:
            action = action.unsqueeze(0)
        
        return {"action": action}
