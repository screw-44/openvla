"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: Torch Tensor of shape (batch_size, C, H, W) or (C, H, W) and datatype torch.float32 with
               values between [0,1], or numpy array of shape (batch_size, H, W, C) or (H, W, C).
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert numpy to torch if needed
    if isinstance(image, np.ndarray):
        # Convert (H, W, C) or (B, H, W, C) to (C, H, W) or (B, C, H, W)
        if image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            expanded_dims = True
        else:
            image = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0
            expanded_dims = False
    else:
        # Assume already torch tensor
        expanded_dims = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            expanded_dims = True

    # Get height and width of crop
    crop_size_ratio = torch.sqrt(torch.tensor(crop_scale)).clamp(0, 1).item()
    
    # Calculate crop size
    _, _, h, w = image.shape
    new_h = int(h * crop_size_ratio)
    new_w = int(w * crop_size_ratio)
    
    # Center crop
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    # Crop all images in batch
    cropped = image[:, :, top:top+new_h, left:left+new_w]
    
    # Resize back to 224x224
    resized = torch.nn.functional.interpolate(cropped, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
    
    # Convert back to original format if needed
    if expanded_dims:
        resized = resized[0]
    
    return resized


def get_vla_action(vla, processor, base_vla_name, obs, task_label, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert PIL Image to torch tensor
        image_np = np.array(image)  # Convert to numpy (H, W, C) uint8
        
        # Crop and resize (function handles conversion internally)
        image_tensor = crop_and_resize(image_np, crop_scale, batch_size)
        
        # Convert back to PIL Image
        # image_tensor is (C, H, W) with values in [0, 1]
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action (returns dict with 'action', 'action_tokens', 'normalized_actions')
    result = vla.predict_action(**inputs, do_sample=False)
    action = result['action']  # Extract continuous action for execution
    return action
